#!/usr/bin/env python3
"""RD + performance comparison: pure-ONNX FP32 vs full-model INT16.

Runs the C end-to-end round-trip (test_cpu_end2end) on real frames for both
model dirs, recording bitrate (bytes), RGB PSNR, and enc+dec wall time.

Usage:
  .venv/bin/python onnx/python/int16_rd_compare.py \
      --frames beauty bosphorus jockey --qps 12 22 32 42 52 --crop 512
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from ptq_dump_calib import load_frame  # noqa: E402

C_BIN = os.path.normpath(os.path.join(REPO, "..", "build", "test_cpu_end2end"))
FP32_DIR = os.path.normpath(os.path.join(REPO, "..", "models_fp32"))
INT16_DIR = os.path.normpath(os.path.join(REPO, "..", "models_int16"))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def bpp(nbytes, h, w):
    return 8.0 * nbytes / (h * w)


def run_one(model_dir, x_path, rec_path, w, h, qp):
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    out = proc.stdout
    if proc.returncode != 0 or "PASS" not in out:
        raise RuntimeError(f"roundtrip failed: {' '.join(cmd)}\n{out}\n{proc.stderr}")
    m = re.search(r"stream_size=(\d+) bytes", out)
    p = re.search(r"RGB PSNR vs input: ([0-9.]+) dB", out)
    nbytes = int(m.group(1))
    psnr_db = float(p.group(1)) if p else float("nan")
    return nbytes, psnr_db, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-dir", default=FP32_DIR)
    ap.add_argument("--int16-dir", default=INT16_DIR)
    ap.add_argument("--frames-dir", default=os.path.normpath(os.path.join(REPO, "..", "rd_frames")))
    ap.add_argument("--frames", nargs="+", default=["beauty", "bosphorus", "jockey"])
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--crop", default="512")
    ap.add_argument("--out", default=os.path.join(REPO, "rd_int16_full"))
    args = ap.parse_args()

    cw = ch = int(args.crop)
    if "x" in str(args.crop).lower():
        cw, ch = (int(v) for v in str(args.crop).lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    rows = []
    print(f"FP32={os.path.basename(args.fp32_dir)} INT16={os.path.basename(args.int16_dir)} "
          f"crop={cw}x{ch} frames={args.frames} qps={args.qps}\n")
    hdr = (f"{'frame':10s} {'qp':>3s} {'bpp32':>8s} {'bpp16':>8s} {'drate%':>7s} "
           f"{'psnr32':>8s} {'psnr16':>8s} {'dPSNR':>7s} {'t32(s)':>7s} {'t16(s)':>7s} {'spd':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for frame in args.frames:
        x = load_frame(args.frames_dir, frame, (cw, ch))
        x_path = os.path.join(args.out, f"{frame}_{cw}x{ch}.npy")
        np.save(x_path, x)
        for qp in args.qps:
            r32 = os.path.join(args.out, f"{frame}_qp{qp}_fp32.npy")
            r16 = os.path.join(args.out, f"{frame}_qp{qp}_int16.npy")
            b32, p32, t32 = run_one(args.fp32_dir, x_path, r32, cw, ch, qp)
            b16, p16, t16 = run_one(args.int16_dir, x_path, r16, cw, ch, qp)
            row = dict(frame=frame, qp=qp, bytes_fp32=b32, bytes_int16=b16,
                       bpp_fp32=bpp(b32, ch, cw), bpp_int16=bpp(b16, ch, cw),
                       psnr_fp32=p32, psnr_int16=p16, t_fp32=t32, t_int16=t16)
            rows.append(row)
            print(f"{frame:10s} {qp:3d} {row['bpp_fp32']:8.4f} {row['bpp_int16']:8.4f} "
                  f"{100*(b16/b32-1):+6.2f}% {p32:8.3f} {p16:8.3f} {p16-p32:+7.3f} "
                  f"{t32:7.2f} {t16:7.2f} {t32/t16:5.2f}x", flush=True)

    with open(os.path.join(args.out, "rd_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== mean over frames (by qp) ===")
    print(f"{'qp':>4} {'bpp32':>8} {'bpp16':>8} {'ratio':>7} {'dPSNR':>7} {'t32':>7} {'t16':>7} {'spd':>6}")
    for qp in args.qps:
        sub = [r for r in rows if r["qp"] == qp]
        b32 = np.mean([r["bpp_fp32"] for r in sub])
        b16 = np.mean([r["bpp_int16"] for r in sub])
        dp = np.mean([r["psnr_int16"] - r["psnr_fp32"] for r in sub])
        t32 = np.mean([r["t_fp32"] for r in sub])
        t16 = np.mean([r["t_int16"] for r in sub])
        print(f"{qp:4d} {b32:8.4f} {b16:8.4f} {b16/b32:7.3f} {dp:+7.3f} "
              f"{t32:7.2f} {t16:7.2f} {t32/t16:6.2f}x")
    # grand averages
    g_drate = np.mean([100*(r["bytes_int16"]/r["bytes_fp32"]-1) for r in rows])
    g_dpsnr = np.mean([r["psnr_int16"]-r["psnr_fp32"] for r in rows])
    g_spd = np.mean([r["t_fp32"]/r["t_int16"] for r in rows])
    print(f"\nOVERALL: mean drate={g_drate:+.2f}%  mean dPSNR={g_dpsnr:+.4f} dB  "
          f"mean speedup={g_spd:.3f}x (t16/t32)")
    print(f"wrote {os.path.join(args.out,'rd_results.json')}")


if __name__ == "__main__":
    raise SystemExit(main())
