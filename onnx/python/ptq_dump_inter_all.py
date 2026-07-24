#!/usr/bin/env python3
"""Dump calibration inputs for ALL 10 inter nets (entropy + reconstruction).

Replicates the C encode dataflow (cpu_inter_pipeline.c) in Python+ORT, saving
each net's call-inputs so fxp_export_entropy_nets.py can quantize them with
real-content activation statistics.

Single-input nets -> <dir>/<net>/sample_N.npy  (4-D [1,C,H,W])
Multi-input  nets -> <dir>/<net>/sample_N.npz  (keys = ONNX input names)

Uses models_fp32/ (all-FP32 pack) so ORT can run every net without the
com.dcvc custom ops.

Usage:
  python ptq_dump_inter_all.py --npy ../akiyo_10frames.npy --qp 32 --frames 0 2 4
"""
from __future__ import annotations
import argparse, os, shutil
import numpy as np
import onnxruntime as ort

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))

YCH, ZCH, DCH, SRCD, RCH = 128, 128, 256, 192, 320
KR, KG, KB = 0.2126, 0.7152, 0.0722  # BT.709 (matches C code)


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = KR * r + KG * g + KB * b
    cb = 0.5 * (b - y) / (1.0 - KB) + 0.5
    cr = 0.5 * (r - y) / (1.0 - KR) + 0.5
    y = np.clip(y, 0, 1); cb = np.clip(cb, 0, 1); cr = np.clip(cr, 0, 1)
    return np.stack([y, cb, cr], 0).astype(np.float32)


def replicate_pad_3(x, H, W):
    Hp = ((H + 63) // 64) * 64; Wp = ((W + 63) // 64) * 64
    if Hp == H and Wp == W:
        return x
    xs = np.clip(np.arange(Wp), 0, W - 1)
    ys = np.clip(np.arange(Hp), 0, H - 1)
    return np.ascontiguousarray(x[:, :][:, ys][:, :, xs])


def pixel_unshuffle_8(ycbcr, Hp, Wp):
    fH, fW = Hp // 8, Wp // 8
    out = np.zeros((SRCD, fH, fW), np.float32)
    for ci in range(3):
        for dy in range(8):
            for dx in range(8):
                oc = ci * 64 + dy * 8 + dx
                out[oc] = ycbcr[ci, dy::8, dx::8]
    return out


def build_masks_2x(yH, yW):
    half = YCH // 2
    m0 = np.zeros((YCH, yH, yW), np.float32)
    m1 = np.zeros((YCH, yH, yW), np.float32)
    for ch in range(YCH):
        second = ch >= half
        for i in range(yH):
            for j in range(yW):
                sp = (i & 1) * 2 + (j & 1)
                p0 = 1 if (sp == 0 or sp == 3) else 0
                val = (1 - p0) if second else p0
                m0[ch, i, j] = val
                m1[ch, i, j] = 1 - val
    return m0, m1


class InterDump:
    def __init__(self, model_dir):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        def mk(name):
            return ort.InferenceSession(os.path.join(model_dir, name + ".onnx"),
                                        so, providers=["CPUExecutionProvider"])
        self.s = {n: mk(n) for n in [
            "inter_feature_adaptor_i", "inter_feature_extractor",
            "inter_encoder", "inter_hyper_enc", "inter_hyper_dec",
            "inter_temporal_prior", "inter_prior_fusion", "inter_spatial_prior",
            "inter_decoder", "recon_generation"]}
        self.qb = {}
        for n in ["q_feature", "q_encoder", "q_decoder", "q_recon"]:
            self.qb[n] = np.load(os.path.join(model_dir, n + ".npy"))

    def _run(self, name, *feeds):
        s = self.s[name]
        ins = s.get_inputs()
        d = {ins[i].name: f for i, f in enumerate(feeds)}
        return s.run(None, d)

    def dump_frame(self, x_rgb, ref_rgb, H, W, qp, out_dir, idx):
        Hp = ((H + 63) // 64) * 64; Wp = ((W + 63) // 64) * 64
        fH, fW = Hp // 8, Wp // 8
        yH, yW = Hp // 16, Wp // 16
        zH, zW = Hp // 64, Wp // 64

        q_feat  = self.qb["q_feature"][qp:qp+1].astype(np.float32).reshape(1, DCH, 1, 1)
        q_enc   = self.qb["q_encoder"][qp:qp+1].astype(np.float32).reshape(1, DCH, 1, 1)
        q_dec   = self.qb["q_decoder"][qp:qp+1].astype(np.float32).reshape(1, DCH, 1, 1)
        q_recon = self.qb["q_recon"][qp:qp+1].astype(np.float32).reshape(1, RCH, 1, 1)

        ref_p = replicate_pad_3(ref_rgb, H, W)
        ref_yc = rgb_to_ycbcr(ref_p)
        ref_un = pixel_unshuffle_8(ref_yc, Hp, Wp)
        feature = self._run("inter_feature_adaptor_i", ref_un[None])[0][0]
        ctx, ctx_t = self._run("inter_feature_extractor", feature[None], q_feat)
        ctx, ctx_t = ctx[0], ctx_t[0]

        x_p = replicate_pad_3(x_rgb, H, W)
        x_yc = rgb_to_ycbcr(x_p)
        x_un = pixel_unshuffle_8(x_yc, Hp, Wp)
        y = self._run("inter_encoder", x_un[None], ctx[None], q_enc)[0][0]

        z = self._run("inter_hyper_enc", y[None])[0][0]
        z_hat = np.round(z).astype(np.float32)

        hier = self._run("inter_hyper_dec", z_hat[None])[0][0]
        temporal = self._run("inter_temporal_prior", ctx_t[None])[0][0]
        pfus_in = np.concatenate([hier, temporal], 0)
        params = self._run("inter_prior_fusion", pfus_in[None])[0][0]

        qdv = np.maximum(params[:YCH], 0.5)
        means0 = params[2*YCH:3*YCH]
        mask0, mask1 = build_masks_2x(yH, yW)
        y_scaled = y / qdv
        yhat_0 = (np.round(np.clip((y_scaled - means0*mask0)*mask0, -128, 127)) + means0*mask0)
        cat_sp = np.concatenate([yhat_0, params], 0)
        sp_out = self._run("inter_spatial_prior", cat_sp[None])[0][0]
        means1 = sp_out[YCH:2*YCH]
        yhat_1 = (np.round(np.clip((y_scaled - means1*mask1)*mask1, -128, 127)) + means1*mask1)
        y_hat = (yhat_0 + yhat_1) * qdv

        fdec = self._run("inter_decoder", y_hat[None], ctx[None], q_dec)[0][0]

        def sn(net, arr):
            d = os.path.join(out_dir, net); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, f"sample_{idx}.npy"),
                    np.ascontiguousarray(arr[None]).astype(np.float32))

        def sz(net, **kw):
            d = os.path.join(out_dir, net); os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, f"sample_{idx}.npz"),
                     **{k: np.ascontiguousarray(v).astype(np.float32) for k, v in kw.items()})

        sn("inter_feature_adaptor_i", ref_un)
        sn("inter_hyper_enc", y)
        sn("inter_hyper_dec", z_hat)
        sn("inter_temporal_prior", ctx_t)
        sn("inter_prior_fusion", pfus_in)
        sn("inter_spatial_prior", cat_sp)
        sz("inter_feature_extractor", in0=feature[None], q_feat=q_feat)
        sz("inter_encoder", in0=x_un[None], ctx=ctx[None], q_enc=q_enc)
        sz("inter_decoder", in0=y_hat[None], ctx=ctx[None], q_dec=q_dec)
        sz("recon_generation", in0=fdec[None], q_recon=q_recon)

        print(f"  sample {idx}: y={y.shape} z={z.shape} fdec={fdec.shape} OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--model-dir", default=os.path.join(ONNX_DIR, "models_fp32"))
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_inter_all"))
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--frames", nargs="*", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(args.calib_dir):
        shutil.rmtree(args.calib_dir)
    os.makedirs(args.calib_dir, exist_ok=True)
    dumper = InterDump(args.model_dir)
    seq = np.load(args.npy)
    N, C, H, W = seq.shape
    frames = args.frames if args.frames else [0, 2, 4]
    for idx, f in enumerate(frames):
        print(f"frame {f} (ref=self):")
        dumper.dump_frame(seq[f], seq[f], H, W, args.qp, args.calib_dir, idx)
    print("done ->", args.calib_dir)


if __name__ == "__main__":
    main()
