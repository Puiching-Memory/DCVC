#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Phase 0: dump per-subgraph golden tensors + CDF assets from PyTorch checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _save_tensor(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = t.detach().cpu().float().numpy()
    np.save(path, arr)


def dump_cdf(model, out_dir: Path, tag: str) -> dict:
    """Export quantized CDF tables after model.update()."""
    model.update(force=True)
    meta = {"tag": tag, "groups": []}
    # BitEstimator / GaussianEncoder expose get_cdf_info via AEHelper
    for name, mod in model.named_modules():
        if hasattr(mod, "get_cdf_info") and hasattr(mod, "_quantized_cdf"):
            if mod._quantized_cdf is None:
                continue
            cdf, cdf_len, offset = mod.get_cdf_info()
            gdir = out_dir / "cdf" / tag / name.replace(".", "_")
            gdir.mkdir(parents=True, exist_ok=True)
            np.save(gdir / "cdf.npy", cdf)
            np.save(gdir / "cdf_length.npy", cdf_len)
            np.save(gdir / "offset.npy", offset)
            meta["groups"].append(
                {
                    "name": name,
                    "cdf_shape": list(np.asarray(cdf).shape),
                    "path": str(gdir.relative_to(out_dir)),
                }
            )
    return meta


def dump_intra_golden(ckpt: Path, out_dir: Path, h: int, w: int, qp: int) -> None:
    from src.models.image_model import DMCI

    net = DMCI().half().cuda().eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "net" in state:
        state = state["net"]
    net.load_state_dict(state, strict=False)
    net.update(force=True)

    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        # Hook encoder output
        y = None

        def hook_enc(_m, _i, o):
            nonlocal y
            y = o.detach()

        hdl = net.enc.register_forward_hook(hook_enc)
        _ = net.compress(x, qp)
        hdl.remove()
        if y is not None:
            _save_tensor(out_dir / "golden" / "intra" / "y.npy", y)
        _save_tensor(out_dir / "golden" / "intra" / "x.npy", x)

    meta = dump_cdf(net, out_dir, "intra")
    (out_dir / "golden" / "intra" / "meta.json").write_text(json.dumps(meta, indent=2))


def dump_inter_golden(ckpt: Path, out_dir: Path, h: int, w: int, qp: int) -> None:
    from src.models.video_model import DMC

    net = DMC().half().cuda().eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "net" in state:
        state = state["net"]
    net.load_state_dict(state, strict=False)
    net.update(force=True)

    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float16)
    # Need DPB: compress I-like first via reset then P
    with torch.no_grad():
        net.clear_dpb()
        # Use feature buffer path similar to test_video
        dpb = {
            "ref_feature": None,
            "ref_frame": x,
            "ref_yuv": None,
        }
        out = net.compress(x, qp, dpb)
        if isinstance(out, dict) and "x_hat" in out:
            _save_tensor(out_dir / "golden" / "inter" / "x_hat.npy", out["x_hat"])
        _save_tensor(out_dir / "golden" / "inter" / "x.npy", x)

    meta = dump_cdf(net, out_dir, "inter")
    (out_dir / "golden" / "inter" / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_image.pth.tar")
    ap.add_argument("--video-ckpt", type=Path, default=ROOT / "checkpoints/cvpr2025_video.pth.tar")
    ap.add_argument("--out", type=Path, default=ROOT / "native/assets")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--skip-cuda", action="store_true", help="Only export CDF if possible on CPU")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not args.skip_cuda:
        print("CUDA required for golden dump (or pass --skip-cuda)", file=sys.stderr)
        sys.exit(1)

    if args.image_ckpt.is_file():
        print(f"Dumping intra golden from {args.image_ckpt}")
        dump_intra_golden(args.image_ckpt, args.out, args.height, args.width, args.qp)
    else:
        print(f"Skip intra: missing {args.image_ckpt}")

    if args.video_ckpt.is_file():
        print(f"Dumping inter golden from {args.video_ckpt}")
        dump_inter_golden(args.video_ckpt, args.out, args.height, args.width, args.qp)
    else:
        print(f"Skip inter: missing {args.video_ckpt}")

    print(f"Done. Assets under {args.out}")


if __name__ == "__main__":
    main()
