#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Generate golden tensors for DCVC-RT native parity testing.

Runs DMCI (I-frame) compress/decompress on GPU and dumps intermediate
tensors + bitstream so the C runtime can be validated against them.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

from src.models.image_model import DMCI
from src.layers.cuda_inference import replicate_pad
from src.utils.common import get_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_image.pth.tar")
    ap.add_argument("--out", type=Path, default=ROOT / "tensorRT/assets/golden")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--qp", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    net = DMCI().to(dev).half().eval()
    net.load_state_dict(get_state_dict(str(args.ckpt)))
    net.update(None)                      # build CDF tables
    net.set_use_two_entropy_coders(True)  # ec_part == 1

    # deterministic YCbCr444 input in [0,1], FP16 as the reference pipeline uses
    g = torch.Generator(dev).manual_seed(1234)
    x = torch.rand((1, 3, args.height, args.width), device=dev, dtype=torch.float16, generator=g)
    x = x * 0.8 + 0.1                     # avoid extreme black/white

    pad_r, pad_b = DMCI.get_padding_size(args.height, args.width, 16)
    x_padded = replicate_pad(x, pad_b, pad_r)
    H, W = x_padded.shape[-2:]

    with torch.no_grad():
        enc = net.compress(x_padded, args.qp)
        bit_stream = enc["bit_stream"]
        x_hat_enc = enc["x_hat"]

        sps = {"sps_id": -1, "height": args.height, "width": args.width,
               "ec_part": 1, "use_ada_i": 0}
        dec = net.decompress(bit_stream, sps, args.qp)
        x_hat_dec = dec["x_hat"]

    # enc/dec reconstruction must match
    max_err = (x_hat_enc.float() - x_hat_dec.float()).abs().max().item()
    crop = x_hat_enc[:, :, :args.height, :args.width].float()
    mse = ((crop - x[:, :args.height, :args.width].float()) ** 2).mean().item()
    psnr = float("inf") if mse == 0 else -10.0 * np.log10(mse)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"i_{args.height}x{args.width}_qp{args.qp}"
    np.save(args.out / f"{tag}_input_fp16.npy",
            x[:, :, :args.height, :args.width].permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
    np.save(args.out / f"{tag}_xhat_enc.npy",
            x_hat_enc[:, :, :args.height, :args.width].permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
    (args.out / f"{tag}_bitstream.bin").write_bytes(bytes(bit_stream))

    meta = {
        "height": args.height, "width": args.width, "pad_h": int(H), "pad_w": int(W),
        "qp": args.qp, "ec_part": 1, "bits": len(bit_stream) * 8,
        "enc_dec_max_abs_err": max_err, "psnr_db": psnr,
        "device": torch.cuda.get_device_name(dev),
    }
    import json
    (args.out / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    assert max_err < 1e-3, f"enc/dec mismatch! max_abs_err={max_err}"


if __name__ == "__main__":
    main()
