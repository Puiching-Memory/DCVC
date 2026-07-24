#!/usr/bin/env python3
"""Dump calibration inputs for the 4 INTER entropy-parameter nets.

Replicates the C encode dataflow (cpu_inter_pipeline.c) in Python+ORT and saves
each entropy net's call-input, so fxp_export_entropy_nets.py can quantize them
with real-content activation statistics (not random), which keeps RD ~1.0x AND
makes the integer path cross-platform bit-exact.

Outputs (default --calib-dir python/calib_inter):
  inter_hyper_dec/0.npy        z_hat      [128, zH, zW]
  inter_temporal_prior/0.npy   ctx_t      [256, yH, yW]
  inter_prior_fusion/0.npy     pfus_in    [384, yH, yW]   (= cat(hier, temporal))
  inter_spatial_prior/0.npy    cat_sp     [512, yH, yW]   (= cat(yhat_acc, params))

Usage:
  python ptq_dump_inter_calib.py --npy ../akiyo_10frames.npy --qp 32
  python ptq_dump_inter_calib.py --npy seq.npy --qp 32 --frames 0 1 2 3
"""
from __future__ import annotations
import argparse, os, shutil
import numpy as np
import onnxruntime as ort

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))

YCH, ZCH, DCH, SRCD = 128, 128, 256, 192


def rgb_to_ycbcr(rgb):
    # rgb: [3,H,W] in [0,1]. BT.601 (the C pipeline matches test_video.py).
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return np.stack([y, cb, cr], 0).astype(np.float32)


def pixel_unshuffle_8(ycbcr, H, W):
    # ycbcr: [3,H,W] -> [192, H/8, W/8], matching C pixel_unshuffle_8.
    fH, fW = H // 8, W // 8
    out = np.zeros((SRCD, fH, fW), np.float32)
    for oc in range(SRCD):
        c = oc // 64; block = oc % 64; dy, dx = block // 8, block % 8
        out[oc] = ycbcr[c, dy::8, dx::8]
    return out


def replicate_pad_3(x, H, W):
    # x: [3,H,W] -> [3,Hp,Wp] with edge replication (F.pad mode="replicate").
    Hp = ((H + 63) // 64) * 64; Wp = ((W + 63) // 64) * 64
    if Hp == H and Wp == W:
        return x
    xs = np.clip(np.arange(Wp), 0, W - 1)
    ys = np.clip(np.arange(Hp), 0, H - 1)
    return np.ascontiguousarray(x[:, :][:, ys][:, :, xs])


class InterDump:
    def __init__(self, model_dir):
        so = ort.SessionOptions()
        def mk(name):
            return ort.InferenceSession(os.path.join(model_dir, name + ".onnx"),
                                        so, providers=["CPUExecutionProvider"])
        self.s = {n: mk(n) for n in [
            "inter_feature_adaptor_i", "inter_feature_extractor",
            "inter_encoder", "inter_hyper_enc", "inter_hyper_dec",
            "inter_temporal_prior", "inter_prior_fusion", "inter_spatial_prior"]}
        self.qb = {}
        for n in ["q_feature", "q_encoder", "q_decoder", "q_recon"]:
            a = np.load(os.path.join(model_dir, n + ".npy"))
            self.qb[n] = a  # [qp_max+1, ch]

    def _run(self, name, *feeds):
        s = self.s[name]
        ins = s.get_inputs()
        d = {ins[i].name: f for i, f in enumerate(feeds)}
        return s.run(None, d)

    def dump_frame(self, x_rgb, ref_rgb, H, W, qp, out_dir):
        Hp = ((H + 63) // 64) * 64; Wp = ((W + 63) // 64) * 64
        fH, fW = Hp // 8, Wp // 8
        yH, yW = Hp // 16, Wp // 16
        zH, zW = Hp // 64, Wp // 64
        q_feat = self.qb["q_feature"][qp:qp+1].astype(np.float32)      # [1,DCH]
        q_enc = self.qb["q_encoder"][qp:qp+1].astype(np.float32)

        ref_rgb_p = replicate_pad_3(ref_rgb, H, W)
        ref = rgb_to_ycbcr(ref_rgb_p)
        ref_un = pixel_unshuffle_8(ref, Hp, Wp)
        feature = self._run("inter_feature_adaptor_i", ref_un[np.newaxis])[0][0]   # [256,fH,fW]
        ctx, ctx_t = self._run("inter_feature_extractor", feature[np.newaxis], q_feat)
        ctx = ctx[0]; ctx_t = ctx_t[0]

        x_rgb_p = replicate_pad_3(x_rgb, H, W)
        xx = rgb_to_ycbcr(x_rgb_p)
        x_un = pixel_unshuffle_8(xx, Hp, Wp)
        y = self._run("inter_encoder", x_un[np.newaxis], ctx[np.newaxis], q_enc)[0][0]  # [128,yH,yW]
        z = self._run("inter_hyper_enc", y[np.newaxis])[0][0]                          # [128,zH,zW]
        z_hat = np.round(z).astype(np.float32)                                         # encode-side z_hat

        # inter_hyper_dec input = z_hat
        hier = self._run("inter_hyper_dec", z_hat[np.newaxis])[0][0]                   # [128,yH,yW]
        # inter_temporal_prior input = ctx_t
        temporal = self._run("inter_temporal_prior", ctx_t[np.newaxis])[0][0]          # [256,yH,yW]
        # inter_prior_fusion input = cat(hier[128], temporal[256]) = [384,yH,yW]
        pfus_in = np.concatenate([hier, temporal], 0)
        params = self._run("inter_prior_fusion", pfus_in[np.newaxis])[0][0]            # [384,yH,yW]

        # spatial prior input = cat(yhat_acc[128], params[384]); yhat_acc pass0 = round((y-means0)*mask0)+means0
        qd = np.maximum(params[:YCH], 0.5)
        scales0 = params[YCH:2*YCH]; means0 = params[2*YCH:3*YCH]
        # checkerboard mask0 (matches C: even spatial positions)
        mask0 = np.zeros((YCH, yH, yW), np.float32)
        for ch in range(YCH):
            for hh in range(yH):
                for ww in range(yW):
                    if (hh % 2) * 2 + (ww % 2) == 0:   # sp==0 in C mask0 (first quarter)
                        mask0[ch, hh, ww] = 1.0
        y_scaled = y / qd
        yhat_acc = (np.round((y_scaled - means0 * mask0) * mask0) + means0 * mask0)
        yhat_acc = yhat_acc * mask0
        cat_sp = np.concatenate([yhat_acc, params], 0)   # [512,yH,yW]

        for name, arr in [("inter_hyper_dec", z_hat), ("inter_temporal_prior", ctx_t),
                          ("inter_prior_fusion", pfus_in), ("inter_spatial_prior", cat_sp)]:
            d = os.path.join(out_dir, name); os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "0.npy"), arr.astype(np.float32))
        print(f"  frame dumped: y={y.shape} z={z.shape} params={params.shape} cat_sp={cat_sp.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="(N,3,H,W) float32 video")
    ap.add_argument("--model-dir", default=os.path.join(ONNX_DIR, "models"))
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_inter"))
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--frames", nargs="*", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(args.calib_dir):
        shutil.rmtree(args.calib_dir)
    os.makedirs(args.calib_dir, exist_ok=True)
    dumper = InterDump(args.model_dir)
    seq = np.load(args.npy)
    N, C, H, W = seq.shape
    assert C == 3, "expected (N,3,H,W)"
    frames = args.frames if args.frames else list(range(min(N, 4)))
    for f in frames:
        print(f"frame {f} (ref=self):")
        dumper.dump_frame(seq[f], seq[f], H, W, args.qp, args.calib_dir)
    print("done ->", args.calib_dir)


if __name__ == "__main__":
    main()
