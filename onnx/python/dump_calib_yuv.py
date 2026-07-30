#!/usr/bin/env python3
"""Dump fxp calibration data from YUV test sequences at native resolution.

For a resolution tier (720p / 1080p), runs the FP32 encode dataflow (intra via
ptq_dump_calib.Encoder, inter via ptq_dump_inter_all.InterDump) on real frames
from datasets/test_sequences and saves every entropy-net call input as
<out>/<net>/sample_N.npy (.npz for multi-input nets) -- the layout
fxp_export_entropy_nets.py --calib-dir expects.

The dynamic FP32 pack (onnx/models_fp32) is used to run the dataflow; the
resulting calibration is resolution-matched to the fixed-size model packs
(models_720p / models_1080p), which keeps the fxp quantization RD-transparent
(random or resolution-mismatched calibration costs real RD, see
experiments/INT16_FULL_RESULTS.md).

Usage:
  python dump_calib_yuv.py 720p  --out calib_720p
  python dump_calib_yuv.py 1080p --out calib_1080p
  python dump_calib_yuv.py 720p --out /tmp/smoke --seqs Johnny --frames 0 --qps 32

Full fixed-size pack recipe (720p example):
  python export_all_models.py --out-dir models_720p_fp32 --height 720 --width 1280
  python export_inter_models.py --out-dir models_720p_fp32 --height 720 --width 1280
  python dump_calib_yuv.py 720p --out calib_720p
  python fxp_export_entropy_nets.py --src-dir models_720p_fp32 \\
      --out-dir models_720p --calib-dir calib_720p
  python fuse_conv_wsrelu.py --src-dir models_720p --out-dir models_720p_fused
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_PY = os.path.dirname(os.path.abspath(__file__))
if REPO_PY not in sys.path:
    sys.path.insert(0, REPO_PY)
ONNX_DIR = os.path.normpath(os.path.join(REPO_PY, ".."))
REPO_ROOT = os.path.normpath(os.path.join(ONNX_DIR, ".."))

from ptq_dump_calib import Encoder        # noqa: E402
from ptq_dump_inter_all import InterDump  # noqa: E402

DEFAULT_MODELS = os.path.join(ONNX_DIR, "models_fp32")
DEFAULT_YUV_ROOT = os.path.join(REPO_ROOT, "datasets", "test_sequences", "YUV")

KR, KG, KB = 0.2126, 0.7152, 0.0722  # BT.709 (matches the C pipeline)

# Resolution tiers: (yuv subdir, sequence name, width, height).
TIERS = {
    "720p": [
        ("HEVC_E", "Johnny_1280x720_60", 1280, 720),
        ("HEVC_E", "KristenAndSara_1280x720_60", 1280, 720),
        ("HEVC_E", "FourPeople_1280x720_60", 1280, 720),
    ],
    "1080p": [
        ("UVG", "Beauty_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
        ("UVG", "Jockey_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
        ("UVG", "Bosphorus_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
    ],
}


def read_yuv420_frame(path, w, h, idx):
    ys, cs = w * h, (w // 2) * (h // 2)
    fs = ys + 2 * cs
    with open(path, "rb") as f:
        f.seek(idx * fs)
        d = f.read(fs)
    if len(d) < fs:
        raise EOFError(f"frame {idx} out of range in {path}")
    y = np.frombuffer(d[:ys], np.uint8).reshape(h, w).astype(np.float32) / 255.0
    u = np.frombuffer(d[ys:ys + cs], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    v = np.frombuffer(d[ys + cs:], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    return y, u, v


def yuv420_to_rgb(y, u, v):
    h, w = y.shape
    uu = np.repeat(np.repeat(u, 2, 0), 2, 1)[:h, :w]
    vv = np.repeat(np.repeat(v, 2, 0), 2, 1)[:h, :w]
    cb = (uu - 0.5) * 2.0 * (1.0 - KB)
    cr = (vv - 0.5) * 2.0 * (1.0 - KR)
    r = y + cr
    b = y + cb
    g = (y - KR * r - KB * b) / KG
    rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)])
    return np.ascontiguousarray(rgb[None].astype(np.float32))


def replicate_pad(x, hp, wp):
    # x: [1,3,H,W] -> [1,3,hp,wp] edge replication (F.pad mode="replicate").
    xs = np.clip(np.arange(wp), 0, x.shape[3] - 1)
    ys = np.clip(np.arange(hp), 0, x.shape[2] - 1)
    return np.ascontiguousarray(x[:, :, ys][:, :, :, xs])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", choices=sorted(TIERS), help="resolution tier preset")
    ap.add_argument("--out", required=True, help="calib output dir (e.g. calib_720p)")
    ap.add_argument("--model-dir", default=DEFAULT_MODELS,
                    help="dynamic FP32 model pack used to run the dataflow")
    ap.add_argument("--yuv-root", default=DEFAULT_YUV_ROOT)
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 100],
                    help="frame indices to sample per sequence")
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="substring filter on sequence names (default: all in tier)")
    args = ap.parse_args()

    seqs = TIERS[args.tier]
    if args.seqs:
        seqs = [s for s in seqs if any(q in s[1] for q in args.seqs)]
    if not seqs:
        ap.error("no sequences selected")

    os.makedirs(args.out, exist_ok=True)
    enc = Encoder(args.model_dir)
    inter = InterDump(args.model_dir)
    counter = [0]

    def dump(net, _call_idx, arr):
        d = os.path.join(args.out, net)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"sample_{counter[0]:04d}.npy"),
                np.ascontiguousarray(arr).astype(np.float32))
        counter[0] += 1

    icnt = 0
    for cls, name, w, h in seqs:
        yuv_path = os.path.join(args.yuv_root, cls, name + ".yuv")
        if not os.path.exists(yuv_path):
            print(f"  skip {name}: {yuv_path} not found")
            continue
        hp = (h + 63) // 64 * 64
        wp = (w + 63) // 64 * 64
        for fidx in args.frames:
            y, u, v = read_yuv420_frame(yuv_path, w, h, fidx)
            x = replicate_pad(yuv420_to_rgb(y, u, v), hp, wp)  # [1,3,hp,wp]
            for qp in args.qps:
                enc.encode_sample(x, qp, dump)
                print(f"intra {name} f{fidx} qp{qp} -> n={counter[0]}", flush=True)
                inter.dump_frame(x[0], x[0], hp, wp, qp, args.out, icnt)
                icnt += 1
                print(f"inter {name} f{fidx} qp{qp}", flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
