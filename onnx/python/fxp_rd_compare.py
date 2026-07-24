#!/usr/bin/env python3
"""RD comparison: FP32 models vs FXP entropy nets on real frames.

Usage:
  uv run python python/fxp_rd_compare.py \\
      --frames-dir rd_frames --frames beauty bosphorus jockey \\
      --qps 12 22 32 42 52 --crop 256 --out python/rd_fxp
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from ptq_dump_calib import load_frame  # noqa: E402

C_BIN = os.path.normpath(os.path.join(REPO, "..", "build", "test_cpu_end2end"))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def bpp(nbytes, h, w):
    return 8.0 * nbytes / (h * w)


def run_one(model_dir, x_path, rec_path, w, h, qp):
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(
            f"roundtrip failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    m = re.search(r"stream_size=(\d+) bytes", proc.stdout)
    md = re.search(r"x_hat max_diff=([0-9.eE+-]+)", proc.stdout)
    return int(m.group(1)), float(md.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-dir", default=os.path.normpath(os.path.join(REPO, "..", "models")))
    ap.add_argument("--fxp-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_fxp")))
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--crop", default="256",
                    help="square size or WxH; multiples of 64")
    ap.add_argument("--out", default=os.path.join(REPO, "rd_fxp"))
    args = ap.parse_args()

    if "x" in str(args.crop).lower():
        cw, ch = (int(v) for v in str(args.crop).lower().split("x"))
    else:
        cw = ch = int(args.crop)

    os.makedirs(args.out, exist_ok=True)
    rows = []

    for frame in args.frames:
        x = load_frame(args.frames_dir, frame, (cw, ch))
        x_path = os.path.join(args.out, f"{frame}_{cw}x{ch}.npy")
        np.save(x_path, x)
        for qp in args.qps:
            rec32 = os.path.join(args.out, f"{frame}_qp{qp}_fp32_rec.npy")
            recfx = os.path.join(args.out, f"{frame}_qp{qp}_fxp_rec.npy")
            print(f"== {frame} qp={qp} FP32 …", flush=True)
            b32, d32 = run_one(args.fp32_dir, x_path, rec32, cw, ch, qp)
            print(f"== {frame} qp={qp} FXP …", flush=True)
            bfx, dfx = run_one(args.fxp_dir, x_path, recfx, cw, ch, qp)
            p32 = psnr(x, np.load(rec32))
            pfx = psnr(x, np.load(recfx))
            row = {
                "frame": frame,
                "qp": qp,
                "bytes_fp32": b32,
                "bytes_fxp": bfx,
                "bpp_fp32": bpp(b32, ch, cw),
                "bpp_fxp": bpp(bfx, ch, cw),
                "psnr_fp32": p32,
                "psnr_fxp": pfx,
                "dpsnr": pfx - p32,
                "byte_ratio": bfx / b32,
                "encdec_maxdiff_fp32": d32,
                "encdec_maxdiff_fxp": dfx,
                "w": cw,
                "h": ch,
            }
            rows.append(row)
            print(
                f"{frame} qp={qp:2d}: bytes {b32:7d} -> {bfx:7d} "
                f"({100 * (bfx / b32 - 1):+6.2f}%), "
                f"PSNR {p32:6.3f} -> {pfx:6.3f} dB ({pfx - p32:+.3f}), "
                f"enc-dec Δ fxp={dfx}",
                flush=True,
            )

    out_json = os.path.join(args.out, "rd_results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_json}")

    # Aggregate by qp (mean over frames)
    print("\n=== mean over frames ===")
    print(f"{'qp':>4} {'bpp32':>8} {'bppFXP':>8} {'ratio':>7} {'PSNR32':>8} {'PSNRFXP':>8} {'dPSNR':>7}")
    for qp in args.qps:
        sub = [r for r in rows if r["qp"] == qp]
        b32 = np.mean([r["bpp_fp32"] for r in sub])
        bfx = np.mean([r["bpp_fxp"] for r in sub])
        p32 = np.mean([r["psnr_fp32"] for r in sub])
        pfx = np.mean([r["psnr_fxp"] for r in sub])
        print(f"{qp:4d} {b32:8.4f} {bfx:8.4f} {bfx/b32:7.3f} {p32:8.3f} {pfx:8.3f} {pfx-p32:+7.3f}")


if __name__ == "__main__":
    main()
