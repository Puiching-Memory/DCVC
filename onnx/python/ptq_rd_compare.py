#!/usr/bin/env python3
"""RD comparison FP32 (onnx/models) vs INT8 PTQ (onnx/models_int8), C end-to-end.

Runs onnx/build/test_cpu_end2end round-trips for a frame x qp matrix on both
model dirs, records the bitstream size and computes RGB PSNR in Python from
the input x.npy (RGB [0,1], NCHW) and the decoder reconstruction rec.npy
(the C test writes the decoded RGB [0,1] frame, see test_cpu_end2end.cpp).

Frames are center-cropped 512x512 (multiple of 64, as required by the C
pipeline). The RD frames differ from the calibration frames on purpose.

Usage:
  python ptq_rd_compare.py [--frames f00050 f00200 f00400] [--qps 22 32 42]
                           [--crop 512] [--out rd_out]
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from ptq_dump_calib import DEFAULT_FRAMES_DIR, load_frame  # noqa: E402

C_BIN = os.path.normpath(os.path.join(REPO, "..", "build", "test_cpu_end2end"))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def run_one(model_dir, x_path, rec_path, w, h, qp):
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(f"roundtrip failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"stream_size=(\d+) bytes", proc.stdout)
    md = re.search(r"x_hat max_diff=([0-9.]+)", proc.stdout)
    return int(m.group(1)), float(md.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.normpath(os.path.join(REPO, "..", "models")))
    ap.add_argument("--int8-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_int8")))
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    ap.add_argument("--frames", nargs="+", default=["f00050", "f00200", "f00400"])
    ap.add_argument("--qps", nargs="+", type=int, default=[22, 32, 42])
    ap.add_argument("--crop", default="512",
                    help="center crop: square size (512) or WxH (1920x1024); multiples of 64")
    ap.add_argument("--out", default=os.path.join(REPO, "rd_out"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cw = ch = args.crop
    if isinstance(args.crop, str) and "x" in args.crop:
        cw, ch = (int(v) for v in args.crop.lower().split("x"))
    else:
        cw = ch = int(args.crop)
    rows = []
    for frame in args.frames:
        x = load_frame(args.frames_dir, frame, (cw, ch))
        x_path = os.path.join(args.out, f"{frame}_{cw}x{ch}.npy")
        np.save(x_path, x)
        for qp in args.qps:
            rec32 = os.path.join(args.out, f"{frame}_{cw}x{ch}_qp{qp}_fp32_rec.npy")
            rec8 = os.path.join(args.out, f"{frame}_{cw}x{ch}_qp{qp}_int8_rec.npy")
            b32, d32 = run_one(args.models_dir, x_path, rec32, cw, ch, qp)
            b8, d8 = run_one(args.int8_dir, x_path, rec8, cw, ch, qp)
            p32 = psnr(x, np.load(rec32))
            p8 = psnr(x, np.load(rec8))
            rows.append((frame, qp, b32, b8, p32, p8, d32, d8))
            print(f"{frame} qp={qp:2d}: bytes {b32:7d} -> {b8:7d} "
                  f"({100.0 * (b8 - b32) / b32:+6.2f}%), "
                  f"PSNR {p32:6.3f} -> {p8:6.3f} dB ({p8 - p32:+6.3f}), "
                  f"enc-dec maxdiff fp32={d32} int8={d8}", flush=True)

    print("\n=== RD summary (INT8 vs FP32) ===")
    print(f"{'frame':10s} {'qp':>3s} {'bytes32':>8s} {'bytes8':>8s} {'dbytes%':>8s} "
          f"{'psnr32':>7s} {'psnr8':>7s} {'dPSNR':>7s}")
    for frame, qp, b32, b8, p32, p8, _, _ in rows:
        print(f"{frame:10s} {qp:3d} {b32:8d} {b8:8d} {100.0 * (b8 - b32) / b32:+7.2f}% "
              f"{p32:7.3f} {p8:7.3f} {p8 - p32:+7.3f}")
    n = len(rows)
    avg_db = sum(100.0 * (b8 - b32) / b32 for _, _, b32, b8, _, _, _, _ in rows) / n
    avg_dp = sum(p8 - p32 for _, _, _, _, p32, p8, _, _ in rows) / n
    avg_b = sum(b8 for _, _, _, b8, _, _, _, _ in rows) / max(1, sum(b32 for _, _, b32, _, _, _, _, _ in rows))
    print(f"{'AVERAGE':10s} {'':3s} {sum(r[2] for r in rows):8d} {sum(r[3] for r in rows):8d} "
          f"{avg_db:+7.2f}% {sum(r[4] for r in rows)/n:7.3f} {sum(r[5] for r in rows)/n:7.3f} {avg_dp:+7.3f}")
    print(f"(overall bytes ratio int8/fp32 = {avg_b:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
