#!/usr/bin/env python3
"""Quick closed-loop RD check: FP32 (models_fp32) vs INT8 QDQ (models_trt_i8).

Runs onnx/build/test_cpu_end2end (full intra encode+decode round-trip, CPU
ORT) for a frame x qp matrix on both model dirs, parses the bitstream size,
and computes RGB PSNR from input x and decoder rec output.

Usage:
  .venv/bin/python onnx/python/trt_i8_rd_check.py [--crop 512] [--qps 17 32 47]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
C_BIN = os.path.normpath(os.path.join(REPO, "..", "build", "test_cpu_end2end"))
FP32_DIR = os.path.normpath(os.path.join(REPO, "..", "models_fp32"))
INT8_DIR = os.path.normpath(os.path.join(REPO, "..", "models_trt_i8"))
FRAMES = ["beauty", "bosphorus", "jockey"]
FRAMES_DIR = os.path.normpath(os.path.join(REPO, "..", "rd_frames"))

sys.path.insert(0, REPO)
from ptq_dump_calib import load_frame  # noqa: E402
from ptq_rd_compare import psnr, run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--qps", nargs="+", type=int, default=[17, 32, 47])
    ap.add_argument("--fp32-dir", default=FP32_DIR)
    ap.add_argument("--int8-dir", default=INT8_DIR)
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="trt_i8_rd_")
    print(f"frame        qp   bits_fp32  bits_int8   dRate%   psnr_fp32  psnr_int8  dPSNR")
    tot = {}
    for frame in FRAMES:
        x = load_frame(FRAMES_DIR, frame, args.crop)
        x_path = os.path.join(tmp, f"{frame}.npy")
        from bench_video import save_npy
        save_npy(x_path, x)
        _, _, h, w = x.shape
        for qp in args.qps:
            row = {}
            for tag, mdir in [("fp32", args.fp32_dir), ("int8", args.int8_dir)]:
                rec_path = os.path.join(tmp, f"rec_{tag}.npy")
                bits, _ = run_one(mdir, x_path, rec_path, w, h, qp)
                rec = np.load(rec_path)
                row[tag] = (bits, psnr(x[0].transpose(1, 2, 0),
                                       rec[0].transpose(1, 2, 0)))
            drate = 100.0 * (row["int8"][0] - row["fp32"][0]) / row["fp32"][0]
            dp = row["int8"][1] - row["fp32"][1]
            print(f"{frame:12s} {qp:3d} {row['fp32'][0]:10d} {row['int8'][0]:11d} "
                  f"{drate:8.2f} {row['fp32'][1]:10.4f} {row['int8'][1]:10.4f} {dp:8.4f}")
            tot.setdefault(qp, []).append((row["fp32"], row["int8"]))

    print("\n== averages per qp ==")
    for qp in args.qps:
        f32 = np.mean([a[0] for a, _ in tot[qp]])
        i8 = np.mean([b[0] for _, b in tot[qp]])
        pf = np.mean([a[1] for a, _ in tot[qp]])
        pi = np.mean([b[1] for _, b in tot[qp]])
        print(f"qp={qp:3d}  dRate={100*(i8-f32)/f32:+.2f}%  dPSNR={pi-pf:+.4f} dB "
              f"(fp32 {pf:.4f} -> int8 {pi:.4f})")


if __name__ == "__main__":
    main()
