#!/usr/bin/env python3
"""Convert a folder of RGB PNG frames into a single (N,3,H,W) float32 npy file.

Usage:
  python png_folder_to_npy.py --folder <dir> --out <npy> [--pattern '*.png']

The PNGs are read in sorted order (numeric if possible), resized to the
first frame's dimensions, and stacked as NCHW [0,1] float32.
"""
import argparse
import os
import re
from PIL import Image
import numpy as np


def natural_sort_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder containing PNG frames")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--pattern", default="*.png", help="glob pattern")
    args = ap.parse_args()

    files = [f for f in os.listdir(args.folder) if f.lower().endswith('.png')]
    files.sort(key=natural_sort_key)
    if not files:
        print("no PNG files found")
        return 1

    imgs = []
    for f in files:
        img = Image.open(os.path.join(args.folder, f)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        imgs.append(arr.transpose(2, 0, 1)[None, ...])
    stack = np.concatenate(imgs, axis=0).astype(np.float32)
    np.save(args.out, stack)
    print(f"wrote {args.out}: shape={stack.shape} dtype={stack.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
