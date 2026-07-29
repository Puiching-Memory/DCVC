#!/usr/bin/env python3
"""Full RD curve on the DCVC-RT paper test set (UVG + HEVC classes).

Reads raw YUV420 8-bit sequences, converts to RGB (BT.709 inverse of the
codec's RGB->YCbCr), pads to 64-multiple, and runs the C end-to-end intra
codec (test_cpu_end2end) on both FP32 and INT16 model dirs across paper-aligned
QP points. Reports per-sequence and averaged bpp/PSNR RD curves.

Usage:
  .venv/bin/python onnx/python/rd_paper_testset.py \
      [--qps 0 13 25 38 50 63] [--frames-per-seq 1] [--seqs ...]
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
ROOT = os.path.normpath(os.path.join(REPO, "..", ".."))
TEST_ROOT = os.path.join(ROOT, "datasets", "test_sequences", "YUV")
C_BIN = os.path.normpath(os.path.join(REPO, "..", "build", "test_cpu_end2end"))
FP32_DIR = os.path.normpath(os.path.join(REPO, "..", "models_fp32"))
INT16_DIR = os.path.normpath(os.path.join(REPO, "..", "models_int16"))

KR, KG, KB = 0.2126, 0.7152, 0.0722

# (class_dir, seq_name, w, h, total_frames)  -- total_frames from test_cfg
SEQS = [
    # UVG 1080p 120fps (600/300 frames)
    ("UVG", "Beauty_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    ("UVG", "Bosphorus_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    ("UVG", "HoneyBee_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    ("UVG", "Jockey_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    ("UVG", "ReadySteadyGo_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    ("UVG", "ShakeNDry_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 300),
    ("UVG", "YachtRide_1920x1080_120fps_420_8bit_YUV", 1920, 1080, 600),
    # HEVC Class E 720p 60fps (600 frames)
    ("HEVC_E", "FourPeople_1280x720_60", 1280, 720, 600),
    ("HEVC_E", "Johnny_1280x720_60", 1280, 720, 600),
    ("HEVC_E", "KristenAndSara_1280x720_60", 1280, 720, 600),
]


def read_yuv420_frame(path, w, h, idx):
    ys, cs = w * h, (w // 2) * (h // 2)
    fs = ys + 2 * cs
    with open(path, "rb") as f:
        f.seek(idx * fs)
        d = f.read(fs)
    if len(d) < fs:
        raise EOFError(f"frame {idx} out of range")
    y = np.frombuffer(d[:ys], np.uint8).reshape(h, w).astype(np.float32) / 255.0
    u = np.frombuffer(d[ys:ys + cs], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    v = np.frombuffer(d[ys + cs:], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    return y, u, v


def yuv420_to_rgb(y, u, v):
    """Inverse BT.709 of the codec's rgb_to_ycbcr (ptq_dump_calib.rgb_to_ycbcr_c)."""
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


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def run_one(model_dir, x_path, rec_path, w, h, qp):
    cmd = [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path]
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    if pr.returncode != 0 or "PASS" not in pr.stdout:
        raise RuntimeError(f"FAIL {' '.join(cmd)}\n{pr.stdout}\n{pr.stderr}")
    nb = int(re.search(r"stream_size=(\d+) bytes", pr.stdout).group(1))
    pp = re.search(r"RGB PSNR vs input: ([0-9.]+) dB", pr.stdout)
    return nb, (float(pp.group(1)) if pp else float("nan")), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32-dir", default=FP32_DIR)
    ap.add_argument("--int16-dir", default=INT16_DIR)
    ap.add_argument("--qps", nargs="+", type=int, default=[0, 13, 25, 38, 50, 63])
    ap.add_argument("--frames-per-seq", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(REPO, "rd_paper"))
    ap.add_argument("--seqs", nargs="+", default=None, help="filter by short name substring")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seqs = SEQS
    if args.seqs:
        seqs = [s for s in SEQS if any(k.lower() in s[1].lower() for k in args.seqs)]

    rows = []
    print(f"test set: {len(seqs)} seqs, qps={args.qps}, frames/seq={args.frames_per_seq}")
    print(f"{'seq':34s} {'cls':6s} {'qp':>3s} {'bpp32':>8s} {'bpp16':>8s} {'drate':>7s} "
          f"{'psnr32':>7s} {'psnr16':>7s} {'dPSNR':>7s} {'t32':>6s} {'t16':>6s}")
    print("-" * 110)
    for cls, name, w, h, nf in seqs:
        yuv_path = os.path.join(TEST_ROOT, cls, name + ".yuv")
        if not os.path.exists(yuv_path):
            print(f"  SKIP {name}: {yuv_path} not found")
            continue
        H2, W2 = ((h + 63) // 64) * 64, ((w + 63) // 64) * 64
        fidxs = [int(round((nf - 1) * i / max(1, args.frames_per_seq)))
                 for i in range(args.frames_per_seq)] or [0]
        for fi, fidx in enumerate(fidxs):
            y, u, v = read_yuv420_frame(yuv_path, w, h, fidx)
            rgb = yuv420_to_rgb(y, u, v)
            pad = np.zeros((1, 3, H2, W2), dtype=np.float32)
            pad[:, :, :h, :w] = rgb
            x_path = os.path.join(args.out, f"{name}_f{fidx}.npy")
            np.save(x_path, np.ascontiguousarray(pad))
            short = name.split("_")[0][:32]
            for qp in args.qps:
                r32 = os.path.join(args.out, f"{name}_f{fidx}_qp{qp}_fp32.npy")
                r16 = os.path.join(args.out, f"{name}_f{fidx}_qp{qp}_i16.npy")
                b32, p32, t32 = run_one(args.fp32_dir, x_path, r32, W2, H2, qp)
                b16, p16, t16 = run_one(args.int16_dir, x_path, r16, W2, H2, qp)
                bpp32 = 8.0 * b32 / (h * w)
                bpp16 = 8.0 * b16 / (h * w)
                rows.append(dict(seq=name, cls=cls, frame=fidx, qp=qp,
                                 bpp_fp32=bpp32, bpp_int16=bpp16,
                                 psnr_fp32=p32, psnr_int16=p16,
                                 bytes_fp32=b32, bytes_int16=b16,
                                 t_fp32=t32, t_int16=t16))
                print(f"{short:34s} {cls:6s} {qp:3d} {bpp32:8.4f} {bpp16:8.4f} "
                      f"{100*(b16/b32-1):+6.2f}% {p32:7.3f} {p16:7.3f} "
                      f"{p16-p32:+7.3f} {t32:6.1f} {t16:6.1f}", flush=True)

    with open(os.path.join(args.out, "rd_paper_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== averaged RD curve (mean over seqs x frames) ===")
    print(f"{'qp':>4} {'bpp32':>8} {'bpp16':>8} {'ratio':>7} {'dPSNR':>7} {'t32':>6} {'t16':>6} {'spd':>6}")
    for qp in args.qps:
        sub = [r for r in rows if r["qp"] == qp]
        b32 = np.mean([r["bpp_fp32"] for r in sub])
        b16 = np.mean([r["bpp_int16"] for r in sub])
        dp = np.mean([r["psnr_int16"] - r["psnr_fp32"] for r in sub])
        t32 = np.mean([r["t_fp32"] for r in sub])
        t16 = np.mean([r["t_int16"] for r in sub])
        print(f"{qp:4d} {b32:8.4f} {b16:8.4f} {b16/b32:7.3f} {dp:+7.3f} {t32:6.1f} {t16:6.1f} {t32/t16:6.2f}x")
    g_dr = np.mean([100*(r["bytes_int16"]/r["bytes_fp32"]-1) for r in rows])
    g_dp = np.mean([r["psnr_int16"]-r["psnr_fp32"] for r in rows])
    g_sp = np.mean([r["t_fp32"]/r["t_int16"] for r in rows])
    print(f"\nOVERALL ({len(rows)} pts): drate={g_dr:+.3f}%  dPSNR={g_dp:+.4f} dB  speed={g_sp:.3f}x")


if __name__ == "__main__":
    raise SystemExit(main())
