#!/usr/bin/env python3
"""Ablation study: INT8 calibration methods + CLE for DCVC-RT entropy nets.

Sweeps calibration methods (minmax, entropy, percentile, distribution) with/without
CLE and with/without output-conv split.  Each config produces a model dir under
onnx/models_abl/, then runs C end-to-end RD comparison (3 frames x 3 QPs at 512x512)
to measure bitrate overhead and PSNR change vs FP32.

Usage:
    python run_ablation.py                  # full sweep
    python run_ablation.py --quick          # reduced frame/qp matrix for speed
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))
FP32_DIR = os.path.normpath(os.path.join(REPO, "..", "models"))
ABL_DIR = os.path.normpath(os.path.join(REPO, "..", "models_abl"))
C_BIN = os.path.normpath(os.path.join(REPO, "..", "build_127", "test_cpu_end2end"))

sys.path.insert(0, REPO)
from ptq_dump_calib import DEFAULT_FRAMES_DIR, load_frame
import numpy as np

# Each config: (label, calibrate_method, cle, split)
CONFIGS = [
    ("minmax_split",          "minmax",       False, True),
    ("entropy_split",         "entropy",      False, True),
    ("pct999_split",          "percentile",   False, True),   # 99.9
    ("pct9999_split",         "percentile",   False, True),   # 99.99
    ("pct99999_split",        "percentile",   False, True),   # 99.999 (ORT default)
    ("distribution_split",    "distribution", False, True),
    ("cle_minmax_split",      "minmax",       True,  True),
    ("cle_pct999_split",      "percentile",   True,  True),
    ("cle_pct9999_split",     "percentile",   True,  True),
    ("pct9999_nosplit",       "percentile",   False, False),
    ("cle_pct9999_nosplit",   "percentile",   True,  False),
]

PERCENTILE_MAP = {
    "pct999":  99.9,
    "pct9999": 99.99,
    "pct99999": 99.999,
    "cle_pct999":  99.9,
    "cle_pct9999": 99.99,
}


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def run_roundtrip(model_dir, x_path, rec_path, w, h, qp):
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(f"roundtrip failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    m = re.search(r"stream_size=(\d+) bytes", proc.stdout)
    return int(m.group(1))


def quantize_config(label, method, cle, split):
    out_dir = os.path.join(ABL_DIR, label)
    cmd = [sys.executable, os.path.join(REPO, "ptq_quantize.py"),
           "--models-dir", FP32_DIR, "--calib-dir", os.path.join(REPO, "calib_data"),
           "--out-dir", out_dir, "--calibrate-method", method]
    if cle:
        cmd.append("--cle")
    if not split:
        cmd.append("--no-split")
    p = label.split("_")[0]
    if "pct" in p and p in PERCENTILE_MAP:
        cmd += ["--percentile", str(PERCENTILE_MAP[p])]
    print(f"\n{'='*60}")
    print(f"Quantizing: {label}")
    print(f"  {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ONNX_DIR, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"  QUANTIZE FAILED:\n{proc.stdout}\n{proc.stderr}")
        return False
    tail = proc.stdout.strip().split("\n")[-1]
    print(f"  {tail}")
    return True


def rd_compare(label, frames, qps, crop=512):
    int8_dir = os.path.join(ABL_DIR, label)
    rd_out = os.path.join(REPO, f"rd_abl_{label}")
    os.makedirs(rd_out, exist_ok=True)
    cw = ch = int(crop)
    results = []
    for frame in frames:
        x = load_frame(DEFAULT_FRAMES_DIR, frame, (cw, ch))
        x_path = os.path.join(rd_out, f"{frame}.npy")
        np.save(x_path, x)
        for qp in qps:
            rec32 = os.path.join(rd_out, f"{frame}_qp{qp}_fp32.npy")
            rec8 = os.path.join(rd_out, f"{frame}_qp{qp}_int8.npy")
            b32 = run_roundtrip(FP32_DIR, x_path, rec32, cw, ch, qp)
            b8 = run_roundtrip(int8_dir, x_path, rec8, cw, ch, qp)
            p32 = psnr(x, np.load(rec32))
            p8 = psnr(x, np.load(rec8))
            results.append((frame, qp, b32, b8, p32, p8))
            print(f"  {frame} qp={qp}: {b32}->{b8} ({100*(b8-b32)/b32:+.1f}%) "
                  f"PSNR {p32:.3f}->{p8:.3f} ({p8-p32:+.3f})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced frame/qp matrix")
    ap.add_argument("--crop", default="512")
    ap.add_argument("--frames", nargs="+", default=["f00050", "f00200", "f00400"])
    ap.add_argument("--qps", nargs="+", type=int, default=[22, 32, 42])
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    frames = ["f00050", "f00200"] if args.quick else args.frames
    qps = [32] if args.quick else args.qps

    all_results = {}
    for label, method, cle, split in CONFIGS:
        int8_dir = os.path.join(ABL_DIR, label)
        if args.skip_existing and os.path.exists(os.path.join(int8_dir, "y_prior_fusion.onnx")):
            print(f"\n[skip quantize] {label} (already exists)")
        else:
            if not quantize_config(label, method, cle, split):
                continue
        try:
            results = rd_compare(label, frames, qps, args.crop)
            all_results[label] = results
        except Exception as e:
            print(f"  RD FAILED: {e}")
            all_results[label] = None

    # Summary
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY (INT8 vs FP32)")
    print(f"{'='*80}")
    print(f"{'config':25s} {'avg dBytes%':>12s} {'ratio':>7s} {'avg dPSNR':>10s} {'n':>3s}")
    print("-" * 60)
    for label, _, _, _ in CONFIGS:
        results = all_results.get(label)
        if not results:
            print(f"{label:25s}  {'FAILED':>12s}")
            continue
        n = len(results)
        avg_db = sum(100.0 * (b8 - b32) / b32 for _, _, b32, b8, _, _ in results) / n
        avg_dp = sum(p8 - p32 for _, _, _, _, p32, p8 in results) / n
        tot_b32 = sum(b32 for _, _, b32, _, _, _ in results)
        tot_b8 = sum(b8 for _, _, _, b8, _, _ in results)
        ratio = tot_b8 / max(1, tot_b32)
        print(f"{label:25s} {avg_db:+11.2f}% {ratio:7.3f} {avg_dp:+9.3f} dB {n:3d}")

    with open(os.path.join(REPO, "ablation_results.json"), "w") as f:
        json.dump({k: v for k, v in all_results.items() if v}, f, indent=2)
    print(f"\nDetailed results saved to {os.path.join(REPO, 'ablation_results.json')}")


if __name__ == "__main__":
    raise SystemExit(main())
