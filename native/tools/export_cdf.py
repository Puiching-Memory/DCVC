#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Export CDF tables and QP module-bank scales to native/assets binary layout."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_qp_bank(path: Path, tensor: torch.Tensor) -> None:
    """tensor: [qp_num, C, 1, 1] FP16/FP32 → header + FP16 rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.detach().cpu().half().contiguous()
    qp_num, ch = int(t.shape[0]), int(t.shape[1])
    flat = t.view(qp_num, ch).numpy()
    with path.open("wb") as f:
        f.write(struct.pack("<ii", qp_num, ch))
        f.write(flat.tobytes())


def export_from_ckpt(image_ckpt: Path, video_ckpt: Path, out: Path) -> None:
    from src.models.image_model import DMCI
    from src.models.video_model import DMC
    from src.utils.common import get_state_dict

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"qp_banks": [], "cdf": []}

    if image_ckpt.is_file():
        net = DMCI()
        net.load_state_dict(get_state_dict(str(image_ckpt)), strict=False)
        net.eval()
        net.update()
        if hasattr(net, "q_scale_enc"):
            p = out / "qp" / "intra_q_scale_enc.bin"
            write_qp_bank(p, net.q_scale_enc)
            manifest["qp_banks"].append(str(p.relative_to(out)))
        if hasattr(net, "q_scale_dec"):
            p = out / "qp" / "intra_q_scale_dec.bin"
            write_qp_bank(p, net.q_scale_dec)
            manifest["qp_banks"].append(str(p.relative_to(out)))
        # CDF
        for name, mod in net.named_modules():
            if hasattr(mod, "get_cdf_info") and getattr(mod, "_quantized_cdf", None) is not None:
                cdf, cdf_len, offset = mod.get_cdf_info()
                g = out / "cdf" / "intra" / name.replace(".", "_")
                g.mkdir(parents=True, exist_ok=True)
                np.save(g / "cdf.npy", cdf)
                np.save(g / "cdf_length.npy", cdf_len)
                np.save(g / "offset.npy", offset)
                manifest["cdf"].append(str(g.relative_to(out)))

    if video_ckpt.is_file():
        net = DMC()
        net.load_state_dict(get_state_dict(str(video_ckpt)), strict=False)
        net.eval()
        net.update()
        for attr in ("q_encoder", "q_decoder", "q_feature", "q_recon"):
            if hasattr(net, attr):
                p = out / "qp" / f"inter_{attr}.bin"
                write_qp_bank(p, getattr(net, attr))
                manifest["qp_banks"].append(str(p.relative_to(out)))
        for name, mod in net.named_modules():
            if hasattr(mod, "get_cdf_info") and getattr(mod, "_quantized_cdf", None) is not None:
                cdf, cdf_len, offset = mod.get_cdf_info()
                g = out / "cdf" / "inter" / name.replace(".", "_")
                g.mkdir(parents=True, exist_ok=True)
                np.save(g / "cdf.npy", cdf)
                np.save(g / "cdf_length.npy", cdf_len)
                np.save(g / "offset.npy", offset)
                manifest["cdf"].append(str(g.relative_to(out)))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_image.pth.tar")
    ap.add_argument("--video-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_video.pth.tar")
    ap.add_argument("--out", type=Path, default=ROOT / "native/assets")
    args = ap.parse_args()
    export_from_ckpt(args.image_ckpt, args.video_ckpt, args.out)


if __name__ == "__main__":
    main()
