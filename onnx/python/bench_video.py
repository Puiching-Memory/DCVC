#!/usr/bin/env python3
"""Drive the pure-CPU DCVC I-frame codec over a real video to measure quality
and rate. Single-frame (intra) codec -> no inter prediction, so we report
per-frame PSNR, bpp, and the I-frame-only rate.

Usage:
  python bench_video.py --video <mp4> --codec-exe <path> --model-dir <dir> \
      --frames 10 --qp 32 --width 960 --height 512

Writes reconstructed PNGs to --out-dir and prints a summary table.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image


def extract_frame(video, index, w, h, tmp):
    """Extract frame `index` from video, resize to (w,h) with Lanczos, return
    float32 NCHW [0,1] RGB and the uint8 HWC image for PSNR/png."""
    # ffmpeg: select one frame. Use -vf scale for resize.
    png = os.path.join(tmp, f"f{index:05d}.png")
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", video,
        "-vf", f"select=eq(n\\,{index}),scale={w}:{h}:flags=lanczos",
        "-vsync", "0", "-frames:v", "1",
        "-pix_fmt", "rgb24", png,
    ]
    subprocess.run(cmd, check=True)
    img = Image.open(png).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # HWC
    f = arr.astype(np.float32) / 255.0
    nchw = f.transpose(2, 0, 1)[None, ...]  # 1x3xHxW
    return nchw, arr, png


def save_npy(path, nchw_f32):
    # Write a minimal float32 NCHW .npy (matches what the C++ reader expects).
    n, c, h, w = nchw_f32.shape
    hdr = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': "
        f"({n}, {c}, {h}, {w}), }}"
    )
    pad = (64 - ((len(hdr) + 10) % 64)) % 64
    hdr = hdr + " " * pad
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY")
        f.write(struct.pack("<BBH", 1, 0, len(hdr)))
        f.write(hdr.encode("ascii"))
        f.write(nchw_f32.astype("<f4").tobytes())


def load_npy_f32(path):
    with open(path, "rb") as f:
        magic = f.read(6)
        assert magic == b"\x93NUMPY", path
        ver, _, hl = struct.unpack("<BBH", f.read(4))
        f.read(hl)
        return np.frombuffer(f.read(), dtype="<f4").copy()


def psnr(a, b):
    """PSNR between two float32 [0,1] RGB images (max value 1.0)."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return 999.0
    return 10.0 * np.log10(1.0 / mse)


def run_frame(codec_exe, model_dir, tmp, idx, nchw, qp):
    H, W = nchw.shape[2], nchw.shape[3]
    x_npy = os.path.join(tmp, f"x_{idx}.npy")
    save_npy(x_npy, nchw)
    bin_ = os.path.join(tmp, f"f{idx}.bin")
    if os.path.exists(bin_):
        os.remove(bin_)
    for s in (bin_ + ".enc.npy",):
        if os.path.exists(s):
            os.remove(s)
    t0 = time.time()
    r = subprocess.run(
        [codec_exe, "--model-dir", model_dir, "--encode", bin_,
         str(H), str(W), str(qp), x_npy],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"encode failed frame {idx}")
    t_enc = time.time() - t0
    # reconstruction = encoder-side x_hat (bit-exact with decoder-side).
    rec = load_npy_f32(bin_ + ".enc.npy").reshape(1, 3, H, W)[0].transpose(1, 2, 0)
    nbytes = os.path.getsize(bin_)
    return rec, nbytes, t_enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--codec-exe", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0,
                    help="frame index stride; 0 => spread evenly")
    ap.add_argument("--out-dir", default="bench_out")
    args = ap.parse_args()

    # The codec replicate-pads H,W up to a multiple of 64 internally, so any
    # positive resolution is accepted here.

    # total frame count for even sampling
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of",
         "default=noprint_wrappers=1:nokey=1", args.video],
        capture_output=True, text=True)
    total = int(probe.stdout.strip()) if probe.stdout.strip().isdigit() else None

    if args.stride:
        idxs = [args.start + i * args.stride for i in range(args.frames)
                if total is None or args.start + i * args.stride < total]
    elif total:
        idxs = [int(round((total - 1) * i / max(1, args.frames - 1)))
                for i in range(args.frames)]
    else:
        idxs = list(range(args.start, args.start + args.frames))

    os.makedirs(args.out_dir, exist_ok=True)
    w, h = args.width, args.height
    pixels = w * h
    print(f"video={os.path.basename(args.video)} total_frames={total} "
          f"sample={idxs} res={w}x{h} qp={args.qp}")
    print(f"{'frame':>5} {'bytes':>9} {'bpp':>7} {'bppp':>7} "
          f"{'PSNR(dB)':>9} {'enc(s)':>7}")

    psnrs, bpps, bppps, times, total_bytes = [], [], [], [], 0
    for i, idx in enumerate(idxs):
        nchw, orig_u8, png = extract_frame(args.video, idx, w, h, args.out_dir)
        rec, nbytes, t_enc = run_frame(
            args.codec_exe, args.model_dir, args.out_dir, idx, nchw, args.qp)
        p = psnr(orig_u8.astype(np.float32) / 255.0, rec)
        bpp = nbytes * 8.0 / pixels                 # bits per pixel
        bppp = nbytes * 8.0 / (pixels)              # alias
        psnrs.append(p); bpps.append(bpp); times.append(t_enc)
        total_bytes += nbytes
        print(f"{idx:5d} {nbytes:9d} {bpp:7.3f} {bppp:7.3f} {p:9.3f} {t_enc:7.2f}")
        # save reconstructed PNG for visual inspection
        rec_u8 = np.clip(rec * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rec_u8).save(
            os.path.join(args.out_dir, f"rec_{idx:05d}.png"))

    print("-" * 52)
    print(f"frames={len(idxs)} qp={args.qp} res={w}x{h}")
    print(f"mean PSNR = {np.mean(psnrs):.3f} dB   (min {np.min(psnrs):.3f}, "
          f"max {np.max(psnrs):.3f})")
    print(f"mean bpp  = {np.mean(bpps):.4f}  ({np.mean(bpps)*pixels/8/1024:.1f} KB/frame)")
    print(f"total bytes = {total_bytes}  ({total_bytes/1024/1024:.2f} MB)")
    print(f"mean encode time = {np.mean(times):.2f} s/frame")
    print(f"reconstructed PNGs + originals in {args.out_dir}/")


if __name__ == "__main__":
    main()
