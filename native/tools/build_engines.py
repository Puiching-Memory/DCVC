#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Build TensorRT engines from ONNX subgraphs (requires tensorrt + CUDA)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


ENGINE_MAP = {
    "intra_hyper_enc.onnx": "intra_hyper",
    "inter_hyper_enc.onnx": "inter_hyper",
    # Additional ONNX slices produced by convert_weights.py / manual export:
    # intra_analysis, intra_synthesis, inter_feature, inter_analysis, ...
}


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # Register DCVC plugins via native library when available
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build failed: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"Wrote {engine_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, default=ROOT / "native/assets")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    onnx_dir = args.assets / "onnx"
    eng_dir = args.assets / "engines"
    built = []
    if not onnx_dir.is_dir():
        print(f"No ONNX dir at {onnx_dir}; run convert_weights.py first", file=sys.stderr)
        sys.exit(1)

    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        print("tensorrt Python package not installed; writing engine stubs only", file=sys.stderr)
        eng_dir.mkdir(parents=True, exist_ok=True)
        # Touch placeholder so CI knows the expected names
        for name in ENGINE_MAP.values():
            p = eng_dir / f"{name}.engine.stub"
            p.write_text("build with TensorRT SDK\n")
            built.append(str(p.relative_to(args.assets)))
        (args.assets / "engines_manifest.json").write_text(json.dumps({"stubs": built}, indent=2))
        return

    for onnx_name, eng_name in ENGINE_MAP.items():
        src = onnx_dir / onnx_name
        if src.is_file():
            build_engine(src, eng_dir / f"{eng_name}.engine", fp16=args.fp16)
            built.append(eng_name)

    (args.assets / "engines_manifest.json").write_text(json.dumps({"engines": built}, indent=2))


if __name__ == "__main__":
    main()
