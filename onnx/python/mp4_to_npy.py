#!/usr/bin/env python3
"""Convert an MP4 video directly into a single (N,3,H,W) float32 npy file.

Usage:
  python mp4_to_npy.py --video <mp4> --out <npy> [--width 1920] [--height 1080]
                       [--start 0] [--frames 100]

Uses ffmpeg to decode/resize. The output is NCHW [0,1] RGB float32, suitable
for test_cpu_inter --encode.
"""
import argparse
import os
import subprocess
import tempfile
from PIL import Image
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--frames", type=int, default=100, help="number of frames")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", args.video,
            "-vf", f"select=gte(n\\,{args.start}),scale={args.width}:{args.height}:flags=lanczos",
            "-vsync", "0", "-frames:v", str(args.frames),
            "-pix_fmt", "rgb24", os.path.join(tmp, "f%05d.png"),
        ]
        subprocess.run(cmd, check=True)
        files = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
        imgs = []
        for f in files:
            img = Image.open(os.path.join(tmp, f)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            imgs.append(arr.transpose(2, 0, 1)[None, ...])
        stack = np.concatenate(imgs, axis=0).astype(np.float32)
    np.save(args.out, stack)
    print(f"wrote {args.out}: shape={stack.shape} dtype={stack.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
