#!/usr/bin/env python3
"""Generate PTQ calibration data for the DCVC-UF intra entropy-parameter nets.

Reproduces the C encoder-side dataflow (onnx/src/cpu_intra_pipeline.c +
onnx/src/cpu_ar_codec.c, encode path only) with onnxruntime + numpy and dumps
the input of every call of the 7 networks that produce the entropy parameters:

    hyper_dec, y_prior_fusion, y_spatial_prior_reduction,
    y_spatial_prior_adaptor_{1,2,3}, y_spatial_prior

The 4-round autoregressive structure (masked quantization + y_hat restore) is
replicated exactly, because the adaptor / spatial-prior inputs depend on
y_hat_so_far. rANS itself is lossless and therefore not simulated; the symbols
fed to it are the same round() values computed here.

Usage:
  python ptq_dump_calib.py [--frames f00001 f00150 f00300 f00450]
                           [--qps 17 32 47] [--crop 512]
                           [--out <calib_dir>] [--max-samples N]
                           [--verify-c-y y.npy --verify-c-params pf.npy]

Cross-check wiring against the C dump hooks:
  DCVC_DUMP_Y=/tmp/y.npy DCVC_DUMP_PARAMS=/tmp/pf.npy \
      ../build/test_cpu_end2end ../models 512 512 32 <x.npy> /tmp/rec.npy
  python ptq_dump_calib.py --frames f00001 --qps 32 --max-samples 1 \
      --verify-c-y /tmp/y.npy --verify-c-params /tmp/pf.npy
"""
import argparse
import json
import os
import re

import numpy as np
import onnxruntime as ort
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = os.path.normpath(os.path.join(REPO, "..", "models"))
DEFAULT_FRAMES_DIR = os.path.normpath(os.path.join(REPO, "..", "..", "video_frames_1080p"))

# Networks whose call inputs are dumped (file names without .onnx).
TARGET_NETS = [
    "hyper_dec",
    "y_prior_fusion",
    "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1",
    "y_spatial_prior_adaptor_2",
    "y_spatial_prior_adaptor_3",
    "y_spatial_prior",
]

# BT.709 coefficients (cpu_intra_pipeline.c)
KR = np.float32(0.2126)
KG = np.float32(0.7152)
KB = np.float32(0.0722)

# cpu_ar_codec.c
K_MASK_PATTERN = [[0, 1, 2, 3], [3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2]]
N_CH = 256
SCALE_MIN = np.float32(0.11)
SCALE_MAX = np.float32(16.0)


def c_round(x):
    """roundf(): round half away from zero, float32 in/out."""
    x = np.asarray(x, dtype=np.float32)
    return (np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))).astype(np.float32)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def rgb_to_ycbcr_c(rgb):
    """C-style RGB->YCbCr (cpu_intra_pipeline.c rgb_to_ycbcr). rgb: [3,H,W] float32."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    y = KR * r + KG * g + KB * b
    cb = np.float32(0.5) * (b - y) / (np.float32(1.0) - KB) + np.float32(0.5)
    cr = np.float32(0.5) * (r - y) / (np.float32(1.0) - KR) + np.float32(0.5)
    out = np.stack([np.clip(y, 0, 1), np.clip(cb, 0, 1), np.clip(cr, 0, 1)]).astype(np.float32)
    return out


def build_masks(yh, yw):
    """masks[m]: [N_CH, yh*yw] float32, matching ws_ensure() in cpu_ar_codec.c."""
    hw = yh * yw
    q = N_CH // 4
    sp = (np.arange(yh)[:, None] % 2 * 2 + np.arange(yw)[None, :] % 2).reshape(-1)
    masks = np.zeros((4, N_CH, hw), dtype=np.float32)
    for m in range(4):
        for ch in range(N_CH):
            quarter = min(ch // q, 3)
            target = K_MASK_PATTERN[m][quarter]
            masks[m, ch] = (sp == target).astype(np.float32)
    return masks


class Encoder:
    """FP32 encoder-side dataflow, mirroring the C pipeline (encode direction)."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sess = {}
        for name in ["intra_analysis_standard", "intra_hyper_enc"] + TARGET_NETS:
            path = os.path.join(model_dir, name + ".onnx")
            self.sess[name] = ort.InferenceSession(path, sess_options=opts,
                                                   providers=["CPUExecutionProvider"])
        qe = np.load(os.path.join(model_dir, "q_scale_enc.npy"))  # [64,368,1,1]
        self.q_scale_enc = qe

    def run(self, name, x):
        return self.sess[name].run(None, {"in0": x})[0]

    def encode_sample(self, x_rgb, qp, dump):
        """Run the encode-side dataflow. x_rgb: [1,3,H,W] float32 (H,W %64==0).

        dump(net_name, call_idx, array) is invoked for every call of a target net.
        Returns dict with y, params_fusion and y_hat for verification.
        """
        _, _, h, w = x_rgb.shape
        assert h % 64 == 0 and w % 64 == 0
        yh, yw = h // 16, w // 16

        x_ycbcr = rgb_to_ycbcr_c(x_rgb[0])[None]
        qenc = self.q_scale_enc[qp:qp + 1].astype(np.float32)  # [1,368,1,1]
        y = self.sess["intra_analysis_standard"].run(
            None, {"in0": x_ycbcr, "in1": qenc})[0].astype(np.float32)
        y = np.clip(y, np.float32(-128.0), np.float32(127.0)).astype(np.float32)

        z = self.run("intra_hyper_enc", y)
        z_hat = np.clip(c_round(z), np.float32(-128.0), np.float32(127.0)).astype(np.float32)

        dump("hyper_dec", 0, z_hat)
        params = self.run("hyper_dec", z_hat)

        dump("y_prior_fusion", 0, params)
        pf = self.run("y_prior_fusion", params)  # [1,514,yh,yw]

        hw = yh * yw
        pfc = pf[0]
        qenc_ar = (sigmoid(pfc[0]) * np.float32(1.5) + np.float32(0.5)).astype(np.float32)
        qdec_ar = (sigmoid(pfc[1]) * np.float32(1.5) + np.float32(0.5)).astype(np.float32)
        scales0 = pfc[2:2 + N_CH].astype(np.float32)
        means0 = pfc[2 + N_CH:2 + 2 * N_CH].astype(np.float32)

        dump("y_spatial_prior_reduction", 0, pf)
        common = self.run("y_spatial_prior_reduction", pf)[0]  # [256,yh,yw]

        yq_full = (y[0] * qenc_ar[None, :]).astype(np.float32)  # [256, yh, yw]
        masks = build_masks(yh, yw)  # [4, 256, hw]

        yhat = np.zeros((N_CH, yh, yw), dtype=np.float32)
        call_idx = 0
        for rnd in range(4):
            if rnd == 0:
                curr_scales, curr_means = scales0, means0
            else:
                cat = np.concatenate([yhat, common], axis=0)[None]  # [1,512,yh,yw]
                dump(f"y_spatial_prior_adaptor_{rnd}", 0, cat)
                sp_out = self.run(f"y_spatial_prior_adaptor_{rnd}", cat)
                dump("y_spatial_prior", call_idx, sp_out)
                call_idx += 1
                out = self.run("y_spatial_prior", sp_out)[0]  # [512,yh,yw]
                curr_scales, curr_means = out[:N_CH], out[N_CH:]

            mask = masks[rnd].reshape(N_CH, yh, yw)
            # process_mask_yq (force_zero_thres disabled, -1)
            fm = mask
            yq = c_round((yq_full - curr_means * fm) * fm)
            yq = np.clip(yq, np.float32(-128.0), np.float32(127.0)).astype(np.float32)
            # sp4x over channel quarters (stride 64)
            yq4 = yq.reshape(4, N_CH // 4, hw)
            yq_w = (yq4[0] + yq4[1] + yq4[2] + yq4[3]).astype(np.float32)  # [64, hw]
            # restore_y_4x + add_inplace
            yq_w_bc = np.tile(yq_w, (4, 1)).reshape(N_CH, yh, yw)
            yhat = (yhat + (yq_w_bc + curr_means) * mask).astype(np.float32)

        yhat = (yhat * qdec_ar[None, :]).astype(np.float32)
        return {"y": y, "params_fusion": pf, "y_hat": yhat[None]}


def load_frame(frames_dir, name, crop):
    """crop: int (square) or (W, H) tuple; 0/None = no crop."""
    path = os.path.join(frames_dir, name if name.endswith(".png") else name + ".png")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
    arr = arr.transpose(2, 0, 1)[None]  # [1,3,H,W]
    if crop:
        cw, ch = (crop, crop) if isinstance(crop, int) else crop
        _, _, h, w = arr.shape
        assert cw % 64 == 0 and ch % 64 == 0, "crop must be a multiple of 64"
        assert h >= ch and w >= cw
        i, j = (h - ch) // 2, (w - cw) // 2
        arr = arr[:, :, i:i + ch, j:j + cw]
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    _, _, h, w = arr.shape
    assert h % 64 == 0 and w % 64 == 0, "frame size must be a multiple of 64"
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--frames", nargs="+",
                    default=["f00001", "f00150", "f00300", "f00450"])
    ap.add_argument("--qps", nargs="+", type=int, default=[17, 32, 47])
    ap.add_argument("--crop", type=int, default=512,
                    help="center crop size (multiple of 64); 0 = full frame")
    ap.add_argument("--out", default=os.path.join(REPO, "calib_data"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = no limit")
    ap.add_argument("--verify-c-y", default=None,
                    help="C DCVC_DUMP_Y npy to cross-check the first sample")
    ap.add_argument("--verify-c-params", default=None,
                    help="C DCVC_DUMP_PARAMS npy to cross-check the first sample")
    args = ap.parse_args()

    enc = Encoder(args.models_dir)

    for net in TARGET_NETS:
        os.makedirs(os.path.join(args.out, net), exist_ok=True)
    os.makedirs(os.path.join(args.out, "x"), exist_ok=True)

    manifest = []
    sample_idx = 0
    checked = False
    for frame in args.frames:
        x = load_frame(args.frames_dir, frame, args.crop)
        x_path = os.path.join(args.out, "x", f"{frame}_crop{args.crop}.npy")
        if not os.path.exists(x_path):
            np.save(x_path, x)
        for qp in args.qps:
            if args.max_samples and sample_idx >= args.max_samples:
                break
            sid = f"s{sample_idx:02d}"
            saved = []

            def dump(net, call_idx, arr, _sid=sid, _saved=saved):
                p = os.path.join(args.out, net, f"{_sid}_c{call_idx}.npy")
                np.save(p, np.ascontiguousarray(arr, dtype=np.float32))
                _saved.append((net, call_idx, list(arr.shape)))

            res = enc.encode_sample(x, qp, dump)
            manifest.append({"sample": sid, "frame": frame, "qp": qp,
                             "crop": args.crop, "dumps": saved})
            print(f"[{sid}] frame={frame} qp={qp}: "
                  + ", ".join(f"{n}[c{c}]{'x'.join(map(str, s))}" for n, c, s in saved))

            if not checked and (args.verify_c_y or args.verify_c_params):
                checked = True
                if args.verify_c_y:
                    cy = np.load(args.verify_c_y)
                    d = np.abs(cy - res["y"])
                    print(f"verify y vs C dump: shape={cy.shape} "
                          f"max_diff={d.max():.6f} mean_diff={d.mean():.8f} "
                          f"bitwise_identical={cy.tobytes() == res['y'].tobytes()}")
                if args.verify_c_params:
                    cp = np.load(args.verify_c_params)
                    d = np.abs(cp - res["params_fusion"])
                    print(f"verify params_fusion vs C dump: shape={cp.shape} "
                          f"max_diff={d.max():.6f} mean_diff={d.mean():.8f} "
                          f"bitwise_identical={cp.tobytes() == res['params_fusion'].tobytes()}")
            sample_idx += 1

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {sample_idx} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
