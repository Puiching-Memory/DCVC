#!/usr/bin/env python3
"""Export DCVC-RK subnets to RKNN-friendly ONNX.

Uses the legacy TorchScript exporter (dynamo=False, opset 17) so F.pixel_shuffle
emits DepthToSpace and F.interpolate emits Resize (both NPU-native / ~free).

Subnets are fused across consecutive NN stages that have no entropy/AR barrier
between them, cutting host↔NPU round-trips for both the ONNX intermediate and
the final .rknn pack.
"""
import argparse
import json
import os
import sys

import torch
from collections import Counter

# `src` (this folder) is a namespace package; run from the DCVC-RK dir or workspace root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # workspace root

import onnx
from src.models.image_model import (
    IntraDecoderRK, IntraAnalysisHyperRK, IntraPriorChainRK,
    YSpatialPriorReductionRK, YSpatialPriorAdaptorRK, YSpatialPriorRK,
    g_ch_enc_dec, g_ch_y as intra_y, g_ch_z as intra_z,
)
from src.models.video_model import (
    InterFeatIRK, InterFeatPRK, InterEncHyperRK, InterPriorChainRK,
    InterDecReconRK, SpatialPriorRK,
    g_ch_src_d, g_ch_y, g_ch_z, g_ch_d, g_ch_recon,
)

OPSET = 17

# 1080p scales: image 1088x1920 ; f=/8=136x240 ; y=/16=68x120 ; z=/64=17x30
# Each entry: (factory, input_shapes, n_outputs)
INTRA = {
    "intra_synthesis": (
        IntraDecoderRK, [(1, intra_y, 68, 120), (1, g_ch_enc_dec, 1, 1)], 1),
    # analysis + hyper_enc -> (y, z)
    "intra_analysis_hyper": (
        IntraAnalysisHyperRK, [(1, 3, 1088, 1920), (1, g_ch_enc_dec, 1, 1)], 2),
    # hyper_dec + prior_fusion -> params_fusion 514ch
    "intra_prior_chain": (
        IntraPriorChainRK, [(1, intra_z, 17, 30)], 1),
    # 4-pass AR (CPU between passes — keep split)
    "y_spatial_prior_reduction": (
        YSpatialPriorReductionRK, [(1, intra_y * 2 + 2, 68, 120)], 1),
    "y_spatial_prior_adaptor_1": (
        YSpatialPriorAdaptorRK, [(1, intra_y * 2, 68, 120)], 1),
    "y_spatial_prior_adaptor_2": (
        YSpatialPriorAdaptorRK, [(1, intra_y * 2, 68, 120)], 1),
    "y_spatial_prior_adaptor_3": (
        YSpatialPriorAdaptorRK, [(1, intra_y * 2, 68, 120)], 1),
    "y_spatial_prior": (
        YSpatialPriorRK, [(1, intra_y * 2, 68, 120)], 1),
}

INTER = {
    # adaptor + extractor -> (memory, ctx)
    "inter_feat_i": (
        InterFeatIRK, [(1, g_ch_src_d, 136, 240)], 2),
    "inter_feat_p": (
        InterFeatPRK, [(1, g_ch_d, 136, 240)], 2),
    # encoder + hyper_enc -> (y, z)
    "inter_enc_hyper": (
        InterEncHyperRK,
        [(1, g_ch_src_d, 136, 240), (1, g_ch_d, 136, 240), (1, g_ch_d, 1, 1)], 2),
    # hyper_dec + temporal + mul/cat + prior_fusion
    "inter_prior_chain": (
        InterPriorChainRK,
        [(1, g_ch_z, 17, 30), (1, g_ch_d, 136, 240), (1, g_ch_d, 1, 1)], 1),
    # AR 2-pass (CPU between) — keep split
    "inter_spatial_prior": (
        SpatialPriorRK, [(1, g_ch_y * 4, 68, 120)], 1),
    # decoder + recon -> (feature, recon_192); feature = next ref
    "inter_dec_recon": (
        InterDecReconRK,
        [(1, g_ch_y, 68, 120), (1, g_ch_d, 136, 240),
         (1, g_ch_d, 1, 1), (1, g_ch_recon, 1, 1)], 2),
}


def _make(factory):
    return factory()


def export(net, shapes, path, n_outputs=1):
    net.eval()
    args = tuple(torch.randn(*s) for s in shapes)
    out_names = [f"out{i}" for i in range(n_outputs)]
    torch.onnx.export(net, args, path, dynamo=False, opset_version=OPSET,
                      input_names=[f"in{i}" for i in range(len(args))],
                      output_names=out_names)
    m = onnx.load(path)
    ops = Counter(n.op_type for n in m.graph.node)
    n_ct = ops.get("ConvTranspose", 0); n_rs = ops.get("Resize", 0); n_dts = ops.get("DepthToSpace", 0)
    n_out = len(m.graph.output)
    print(f"  {os.path.basename(path):32s} outs={n_out} ConvTranspose={n_ct} "
          f"Resize={n_rs} DepthToSpace={n_dts}")
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "onnx", "models_rk"))
    ap.add_argument("--subnets", nargs="*", default=None)
    ap.add_argument("--ffn-expansion", type=int, default=2)
    ap.add_argument("--inter", action="store_true", help="export inter (P-frame) subnets")
    ap.add_argument("--all", action="store_true", help="export intra + inter")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    registry = INTER if args.inter else (INTRA if not args.all else {**INTRA, **INTER})
    label = "INTER" if args.inter else ("ALL" if args.all else "INTRA")
    print(f"DCVC-RK {label} export -> {args.out_dir}  (opset {OPSET}, fused subnets)")
    shapes_manifest = {}
    for name in (args.subnets or list(registry)):
        if name not in registry:
            print(f"  skip unknown: {name}"); continue
        factory, shapes, n_out = registry[name]
        export(_make(factory), shapes, os.path.join(args.out_dir, name + ".onnx"), n_out)
        shapes_manifest[name] = [list(s) for s in shapes]
    # Single source of truth for input shapes: consumed verbatim by build_rknn.py.
    with open(os.path.join(args.out_dir, "shapes.json"), "w") as f:
        json.dump({"opset": OPSET, "fused": True, "subnets": shapes_manifest}, f, indent=2)
    print(f"done ({len(shapes_manifest)} subnets)")


if __name__ == "__main__":
    main()
