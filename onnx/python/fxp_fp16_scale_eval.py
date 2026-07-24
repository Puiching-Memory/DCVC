#!/usr/bin/env python3
"""Evaluate float16 quantization on the rANS scale path (0-error tolerance).

FXP entropy nets emit float32 scales; before rANS those scales are mapped to
CDF indexes via the integer Q18→Q12→table rule (`fxp_scale_index.c`).
Any index flip between encode and decode desyncs rANS — this path has
**zero error tolerance**.

This script:
  1) Dumps `params_fusion` from the FXP CPU codec on real frames
  2) Round-trips scale channels through IEEE float16
  3) Reports value error + integer scale-index agreement (must be 100%)
  4) Times encode/decode for inference performance
  5) Summarizes existing FXP↔FP32 RD (`rd_fxp_real`) as bitrate/PSNR context

Usage:
  uv run python python/fxp_fp16_scale_eval.py \\
      --frames-dir ../rd_frames --frames beauty bosphorus jockey \\
      --qps 12 22 32 42 52 --crop 256
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.normpath(os.path.join(REPO, ".."))
sys.path.insert(0, REPO)
from ptq_dump_calib import load_frame  # noqa: E402

C_BIN = os.path.join(ONNX, "build", "test_cpu_end2end")

# Q12 log-spaced table from fxp_scale_index.c
_SCALE_TABLE_Q12 = np.array(
    [
        451, 469, 487, 507, 527, 548, 570, 593,
        617, 641, 667, 694, 721, 750, 780, 811,
        844, 878, 913, 949, 987, 1027, 1068, 1110,
        1155, 1201, 1249, 1299, 1351, 1405, 1461, 1519,
        1580, 1643, 1709, 1777, 1848, 1922, 1999, 2079,
        2162, 2249, 2339, 2432, 2530, 2631, 2736, 2845,
        2959, 3077, 3201, 3329, 3462, 3600, 3744, 3894,
        4049, 4211, 4380, 4555, 4737, 4927, 5124, 5328,
        5542, 5763, 5994, 6233, 6483, 6742, 7011, 7292,
        7583, 7887, 8202, 8530, 8871, 9226, 9595, 9979,
        10378, 10793, 11224, 11673, 12140, 12625, 13130, 13655,
        14202, 14769, 15360, 15974, 16613, 17278, 17968, 18687,
        19434, 20212, 21020, 21860, 22735, 23644, 24589, 25573,
        26595, 27659, 28765, 29915, 31112, 32356, 33650, 34995,
        36395, 37850, 39364, 40938, 42575, 44278, 46049, 47890,
        49805, 51797, 53868, 56023, 58263, 60593, 63016, 65536,
    ],
    dtype=np.int32,
)
_Q18_ONE = 262144.0
_SCALE_MIN_Q12 = 451
_SCALE_MAX_Q12 = 65536


def scale_index_i(scales: np.ndarray) -> np.ndarray:
    """Mirror of dcvc_scale_float_to_index (integer path)."""
    s = np.asarray(scales, dtype=np.float32).ravel()
    # half-away-from-zero → Q18
    v = s.astype(np.float64) * _Q18_ONE
    q18 = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)
    q18 = np.clip(q18, -2147483648, 2147483647).astype(np.int32)
    # >>6 round-to-nearest (ties away via +32); scales >= 0
    q12 = ((q18.astype(np.int64) + 32) >> 6).astype(np.int32)
    q12 = np.clip(q12, _SCALE_MIN_Q12, _SCALE_MAX_Q12)
    # largest i with table[i] <= q12
    idx = np.searchsorted(_SCALE_TABLE_Q12, q12, side="right") - 1
    return np.clip(idx, 0, 127).astype(np.int32)


def fp16_roundtrip(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def encode_dump(model_dir, x_path, rec_path, dump_path, h, w, qp):
    env = os.environ.copy()
    env["DCVC_DUMP_PARAMS"] = dump_path
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(
            f"encode failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    m = re.search(r"stream_size=(\d+) bytes", proc.stdout)
    md = re.search(r"x_hat max_diff=([0-9.eE+-]+)", proc.stdout)
    return int(m.group(1)), float(md.group(1)), dt


def analyze_scales(params: np.ndarray, n_ch: int = 256):
    """params: [1, 2N+2, H, W] or [2N+2, H, W]. scales = channels [2, 2+N)."""
    p = np.asarray(params, dtype=np.float32)
    if p.ndim == 4:
        p = p[0]
    scales = p[2 : 2 + n_ch]
    s32 = scales.astype(np.float32)
    s16 = fp16_roundtrip(s32)
    err = np.abs(s16.astype(np.float64) - s32.astype(np.float64))
    i32 = scale_index_i(s32)
    i16 = scale_index_i(s16)
    mism = int(np.sum(i32 != i16))
    return {
        "n": int(s32.size),
        "scale_max_abs_err": float(err.max()),
        "scale_mean_abs_err": float(err.mean()),
        "scale_rms_err": float(np.sqrt(np.mean(err ** 2))),
        "index_mismatch": mism,
        "index_agree_pct": 100.0 * (1.0 - mism / s32.size),
        "index_max_abs_d": int(np.max(np.abs(i32.astype(np.int32) - i16))) if mism else 0,
        "frac_clamped_lo": float(np.mean(s32 <= 0.11)),
        "scale_min": float(s32.min()),
        "scale_max": float(s32.max()),
        "scale_p50": float(np.median(s32)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ONNX, "models"))
    ap.add_argument("--frames-dir", default=os.path.join(ONNX, "rd_frames"))
    ap.add_argument("--frames", nargs="+", default=["beauty", "bosphorus", "jockey"])
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--crop", default="256")
    ap.add_argument("--out", default=os.path.join(REPO, "fp16_scale_eval"))
    ap.add_argument("--rd-json", default=os.path.join(REPO, "rd_fxp_real", "rd_results.json"))
    args = ap.parse_args()

    if "x" in str(args.crop).lower():
        cw, ch = (int(v) for v in str(args.crop).lower().split("x"))
    else:
        cw = ch = int(args.crop)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    total_mism = 0
    total_n = 0

    print("=" * 72)
    print(" FXP float16 scale-path eval (rANS pre-index, 0-error tolerance)")
    print("=" * 72)
    print(f" model={args.model_dir}  crop={cw}x{ch}  frames={args.frames}  qps={args.qps}")
    print()

    for frame in args.frames:
        x = load_frame(args.frames_dir, frame, (cw, ch))
        x_path = os.path.join(args.out, f"{frame}_{cw}x{ch}.npy")
        np.save(x_path, x)
        for qp in args.qps:
            rec = os.path.join(args.out, f"{frame}_qp{qp}_rec.npy")
            dump = os.path.join(args.out, f"{frame}_qp{qp}_params.npy")
            nbytes, maxdiff, dt = encode_dump(
                args.model_dir, x_path, rec, dump, ch, cw, qp
            )
            params = np.load(dump)
            # dump is NCHW without batch in writer? check shape
            stats = analyze_scales(params)
            p = psnr(x, np.load(rec))
            row = {
                "frame": frame,
                "qp": qp,
                "bytes": nbytes,
                "bpp": 8.0 * nbytes / (ch * cw),
                "psnr": p,
                "encdec_maxdiff": maxdiff,
                "wall_s": dt,
                **stats,
            }
            rows.append(row)
            total_mism += stats["index_mismatch"]
            total_n += stats["n"]
            flag = "OK" if stats["index_mismatch"] == 0 else "FAIL"
            print(
                f"{frame:10s} qp={qp:2d}: bytes={nbytes:6d} bpp={row['bpp']:.4f} "
                f"PSNR={p:6.3f} enc↔decΔ={maxdiff:.0e}  "
                f"fp16|err|max={stats['scale_max_abs_err']:.3e} mean={stats['scale_mean_abs_err']:.3e}  "
                f"idx_agree={stats['index_agree_pct']:.4f}% mism={stats['index_mismatch']}  "
                f"t={dt*1e3:.1f}ms [{flag}]",
                flush=True,
            )

    out_json = os.path.join(args.out, "fp16_scale_results.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print()
    print("--- float16 on scales (integer index path) ---")
    print(f" total scale elements: {total_n}")
    print(f" total index mismatches: {total_mism}")
    agree = 100.0 * (1.0 - total_mism / max(total_n, 1))
    print(f" overall index agree: {agree:.6f}%")
    max_err = max(r["scale_max_abs_err"] for r in rows)
    mean_err = float(np.mean([r["scale_mean_abs_err"] for r in rows]))
    print(f" scale |err| max={max_err:.6e}  mean(over runs)={mean_err:.6e}")
    if total_mism == 0:
        print(" VERDICT: PASS — float16 round-trip on scales flips 0 CDF indexes")
        print("          → 0 bitrate overhead from fp16-on-scale; rANS stays locked")
    else:
        print(" VERDICT: FAIL — float16 flips scale indexes (rANS 0-error violated)")
        print(f"          → unsafe for enc/dec if only one side uses fp16 scales")

    # Inference perf summary
    print()
    print("--- inference (FXP end-to-end wall, includes ORT+rANS) ---")
    times = [r["wall_s"] * 1e3 for r in rows]
    print(f" mean wall {np.mean(times):.1f} ms/frame  "
          f"min {np.min(times):.1f}  max {np.max(times):.1f}  "
          f"~{1000/np.mean(times):.2f} fps  ({cw}x{ch})")

    # Dense synthetic probe (worst-case near CDF boundaries)
    print()
    print("--- synthetic linspace / boundary probe (fp16 vs fp32 indexes) ---")
    s = np.linspace(0.05, 20.0, 200000, dtype=np.float32)
    mism = int(np.sum(scale_index_i(s) != scale_index_i(fp16_roundtrip(s))))
    print(f" linspace[0.05,20] n={s.size}: mism={mism} "
          f"agree={100*(1-mism/s.size):.4f}%  (FAIL if >0 under 0-tol)")
    table = _SCALE_TABLE_Q12.astype(np.float32) / 4096.0
    vals = []
    for t in table:
        vals.append(np.linspace(t - 2e-3, t + 2e-3, 401, dtype=np.float32))
    sb = np.concatenate(vals)
    mism_b = int(np.sum(scale_index_i(sb) != scale_index_i(fp16_roundtrip(sb))))
    print(f" table-boundary ±2e-3 n={sb.size}: mism={mism_b} "
          f"agree={100*(1-mism_b/sb.size):.4f}%")

    # RD: same FXP models, scales indexed as fp32 vs fp16 (DCVC_SCALE_FP16=1)
    print()
    print("--- RD: FXP + fp32-scale-index vs FXP + fp16-scale-index ---")
    print(" (both enc&dec share the flag → sync OK; measures bitrate/PSNR delta)")
    rd_rows = []
    for frame in args.frames:
        x_path = os.path.join(args.out, f"{frame}_{cw}x{ch}.npy")
        x = np.load(x_path)
        for qp in args.qps:
            rec0 = os.path.join(args.out, f"{frame}_qp{qp}_f32idx_rec.npy")
            rec1 = os.path.join(args.out, f"{frame}_qp{qp}_f16idx_rec.npy")
            b0, d0, t0 = encode_dump(args.model_dir, x_path, rec0,
                                     os.path.join(args.out, "_unused0.npy"), ch, cw, qp)
            env1 = os.environ.copy()
            env1["DCVC_SCALE_FP16"] = "1"
            # re-encode with fp16 scale path
            cmd = [C_BIN, args.model_dir, str(ch), str(cw), str(qp), x_path, rec1]
            t_a = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env1)
            t1 = time.perf_counter() - t_a
            if proc.returncode != 0 or "PASS" not in proc.stdout:
                raise RuntimeError(f"fp16-scale encode failed\n{proc.stdout}\n{proc.stderr}")
            b1 = int(re.search(r"stream_size=(\d+) bytes", proc.stdout).group(1))
            d1 = float(re.search(r"x_hat max_diff=([0-9.eE+-]+)", proc.stdout).group(1))
            p0 = psnr(x, np.load(rec0))
            p1 = psnr(x, np.load(rec1))
            row = {
                "frame": frame, "qp": qp,
                "bytes_f32idx": b0, "bytes_f16idx": b1,
                "bpp_f32idx": 8.0 * b0 / (ch * cw),
                "bpp_f16idx": 8.0 * b1 / (ch * cw),
                "byte_ratio": b1 / max(b0, 1),
                "psnr_f32idx": p0, "psnr_f16idx": p1, "dpsnr": p1 - p0,
                "encdec_maxdiff_f32idx": d0, "encdec_maxdiff_f16idx": d1,
                "wall_ms_f32idx": t0 * 1e3, "wall_ms_f16idx": t1 * 1e3,
            }
            rd_rows.append(row)
            print(
                f"{frame:10s} qp={qp:2d}: bytes {b0:6d} -> {b1:6d} "
                f"({100*(b1/max(b0,1)-1):+6.2f}%)  "
                f"PSNR {p0:6.3f} -> {p1:6.3f} ({p1-p0:+.4f})  "
                f"enc↔decΔ f16={d1:.0e}",
                flush=True,
            )

    rd_out = os.path.join(args.out, "rd_fp16_scale.json")
    with open(rd_out, "w") as f:
        json.dump(rd_rows, f, indent=2)
    print()
    print(f"{'qp':>4} {'bpp_f32':>8} {'bpp_f16':>8} {'ratio':>7} {'dPSNR':>8}")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rd_rows:
        by[r["qp"]].append(r)
    for qp in sorted(by):
        s = by[qp]
        b0 = np.mean([x["bpp_f32idx"] for x in s])
        b1 = np.mean([x["bpp_f16idx"] for x in s])
        dps = np.mean([x["dpsnr"] for x in s])
        print(f"{qp:4d} {b0:8.4f} {b1:8.4f} {b1/b0:7.4f} {dps:+8.4f}")
    print(f" mean byte_ratio={np.mean([x['byte_ratio'] for x in rd_rows]):.6f}  "
          f"max enc↔decΔ (fp16-idx)={max(x['encdec_maxdiff_f16idx'] for x in rd_rows):.1e}")

    # Existing FXP vs FP32 RD context
    if os.path.isfile(args.rd_json):
        rd = json.load(open(args.rd_json))
        print()
        print(f"--- FXP vs FP32 RD context ({args.rd_json}) ---")
        print(f"{'qp':>4} {'bpp32':>8} {'bppFXP':>8} {'ratio':>7} {'dPSNR':>8} {'encdecΔ':>8}")
        by = defaultdict(list)
        for r in rd:
            by[r["qp"]].append(r)
        for qp in sorted(by):
            s = by[qp]
            b32 = np.mean([x["bpp_fp32"] for x in s])
            bfx = np.mean([x["bpp_fxp"] for x in s])
            dps = np.mean([x["dpsnr"] for x in s])
            md = max(x["encdec_maxdiff_fxp"] for x in s)
            print(f"{qp:4d} {b32:8.4f} {bfx:8.4f} {bfx/b32:7.4f} {dps:+8.4f} {md:8.1e}")
        print(f" mean byte_ratio={np.mean([x['byte_ratio'] for x in rd]):.6f}  "
              f"max enc↔dec FXP Δ={max(x['encdec_maxdiff_fxp'] for x in rd):.1e}")

    print(f"\nwrote {out_json}")
    print(f"wrote {rd_out}")
    # Real-content index agree is the primary 0-tol gate; synthetic may FAIL.
    return 0 if total_mism == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
