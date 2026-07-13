#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Convert PyTorch checkpoints to ONNX subgraphs + weight blobs for TensorRT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export_module_onnx(module: nn.Module, example_inputs: tuple, path: Path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(
        module,
        example_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote {path}")


def convert_intra(ckpt: Path, out: Path, h: int, w: int) -> dict:
    from src.models.image_model import DMCI
    from src.utils.common import get_state_dict

    net = DMCI().eval()
    net.load_state_dict(get_state_dict(str(ckpt)), strict=False)
    # Export synthesis-ish leaf modules that are static graphs
    # Encoder expects pixel-unshuffled input internally; export enc submodule if present
    meta = {"subgraphs": []}
    if hasattr(net, "enc"):
        # Many blocks use custom DepthConv — ONNX may fall back; record attempt
        x = torch.randn(1, 3, h, w)
        try:
            # Full forward not exported; export hyper_enc as example static slice
            if hasattr(net, "hyper_enc"):
                y = torch.randn(1, 256, h // 16, w // 16)
                p = out / "onnx" / "intra_hyper_enc.onnx"
                export_module_onnx(net.hyper_enc, (y,), p, ["y"], ["z"])
                meta["subgraphs"].append(str(p.relative_to(out)))
        except Exception as e:  # noqa: BLE001
            meta["errors"] = meta.get("errors", []) + [f"intra hyper_enc: {e}"]
    # Dump raw state_dict keys for Plugin weight packing
    sd = net.state_dict()
    torch.save({k: v.half().cpu() for k, v in sd.items()}, out / "weights" / "intra_fp16.pth")
    meta["weights"] = "weights/intra_fp16.pth"
    return meta


def convert_inter(ckpt: Path, out: Path, h: int, w: int) -> dict:
    from src.models.video_model import DMC
    from src.utils.common import get_state_dict

    net = DMC().eval()
    net.load_state_dict(get_state_dict(str(ckpt)), strict=False)
    meta = {"subgraphs": []}
    try:
        if hasattr(net, "hyper_encoder"):
            y = torch.randn(1, 128, h // 16, w // 16)
            p = out / "onnx" / "inter_hyper_enc.onnx"
            export_module_onnx(net.hyper_encoder, (y,), p, ["y"], ["z"])
            meta["subgraphs"].append(str(p.relative_to(out)))
    except Exception as e:  # noqa: BLE001
        meta["errors"] = meta.get("errors", []) + [f"inter hyper_enc: {e}"]
    sd = net.state_dict()
    (out / "weights").mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half().cpu() for k, v in sd.items()}, out / "weights" / "inter_fp16.pth")
    meta["weights"] = "weights/inter_fp16.pth"
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_image.pth.tar")
    ap.add_argument("--video-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_video.pth.tar")
    ap.add_argument("--out", type=Path, default=ROOT / "native/assets")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights").mkdir(parents=True, exist_ok=True)
    (args.out / "onnx").mkdir(parents=True, exist_ok=True)

    manifest = {}
    if args.image_ckpt.is_file():
        manifest["intra"] = convert_intra(args.image_ckpt, args.out, args.height, args.width)
    if args.video_ckpt.is_file():
        manifest["inter"] = convert_inter(args.video_ckpt, args.out, args.height, args.width)
    (args.out / "convert_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
