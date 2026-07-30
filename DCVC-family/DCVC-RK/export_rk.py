#!/usr/bin/env python3
"""Export DCVC-RK subnets to RKNN-friendly ONNX.

Uses the legacy TorchScript exporter (dynamo=False, opset 17) so F.pixel_shuffle
emits DepthToSpace and F.interpolate emits Resize (both NPU-native / ~free).
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
    IntraDecoderRK, IntraEncoderRK, IntraHyperEncoderRK, IntraHyperDecoderRK,
    IntraPriorFusionRK,
    YSpatialPriorReductionRK, YSpatialPriorAdaptorRK, YSpatialPriorRK,
    g_ch_enc_dec, g_ch_y as intra_y, g_ch_z as intra_z,
)
from src.models.video_model import (
    FeatureAdaptorIRK, FeatureAdaptorPRK, FeatureExtractorRK, EncoderRK,
    HyperEncoderRK, HyperDecoderRK, TemporalPriorEncoderRK, PriorFusionRK,
    SpatialPriorRK, DecoderRK, ReconGenerationRK,
    g_ch_src_d, g_ch_y, g_ch_z, g_ch_d, g_ch_recon,
)

OPSET = 17

# 1080p scales: image 1088x1920 ; f=/8=136x240 ; y=/16=68x120 ; z=/64=17x30
INTRA = {
    "intra_synthesis":         (IntraDecoderRK,      [(1, intra_y, 68, 120), (1, g_ch_enc_dec, 1, 1)]),
    "intra_analysis_standard": (IntraEncoderRK,      [(1, 3, 1088, 1920), (1, g_ch_enc_dec, 1, 1)]),
    "intra_hyper_enc":         (IntraHyperEncoderRK, [(1, intra_y, 68, 120)]),
    "hyper_dec":               (IntraHyperDecoderRK, [(1, intra_z, 17, 30)]),
    # Runtime filename MUST be y_prior_fusion.onnx/.rknn (loaded by cpu_intra_pipeline.c).
    # Input is hyper_dec output (256ch @ y-res 68x120); output 2*N+2=514ch = mask+scales+means.
    "y_prior_fusion":          (IntraPriorFusionRK,  [(1, intra_y, 68, 120)]),
    # 4-pass AR spatial-prior chain (loaded by cpu_ar_codec.c). intra-only: the
    # video checkpoint has NO y_spatial_prior_* weights, so these come from the
    # image checkpoint and are image-specific. W=512=2*g_ch_y, y-res 68x120.
    "y_spatial_prior_reduction": (YSpatialPriorReductionRK, [(1, intra_y * 2 + 2, 68, 120)]),
    "y_spatial_prior_adaptor_1": (YSpatialPriorAdaptorRK,   [(1, intra_y * 2, 68, 120)]),
    "y_spatial_prior_adaptor_2": (YSpatialPriorAdaptorRK,   [(1, intra_y * 2, 68, 120)]),
    "y_spatial_prior_adaptor_3": (YSpatialPriorAdaptorRK,   [(1, intra_y * 2, 68, 120)]),
    "y_spatial_prior":          (YSpatialPriorRK,           [(1, intra_y * 2, 68, 120)]),
}

INTER = {
    "inter_feature_adaptor_i":  (FeatureAdaptorIRK,       [(1, g_ch_src_d, 136, 240)]),
    "inter_feature_adaptor_p":  (FeatureAdaptorPRK,       [(1, g_ch_d, 136, 240)]),
    "inter_feature_extractor":  (FeatureExtractorRK,      [(1, g_ch_d, 136, 240)]),
    "inter_encoder":            (EncoderRK,               [(1, g_ch_src_d, 136, 240), (1, g_ch_d, 136, 240), (1, g_ch_d, 1, 1)]),
    "inter_hyper_enc":          (HyperEncoderRK,          [(1, g_ch_y, 68, 120)]),
    "inter_hyper_dec":          (HyperDecoderRK,          [(1, g_ch_z, 17, 30)]),
    "inter_temporal_prior":     (TemporalPriorEncoderRK,  [(1, g_ch_d, 136, 240)]),
    "inter_prior_fusion":       (PriorFusionRK,           [(1, g_ch_y * 3, 68, 120)]),
    "inter_spatial_prior":      (SpatialPriorRK,          [(1, g_ch_y * 4, 68, 120)]),
    "inter_decoder":            (DecoderRK,               [(1, g_ch_y, 68, 120), (1, g_ch_d, 136, 240), (1, g_ch_d, 1, 1)]),
    "recon_generation":         (ReconGenerationRK,       [(1, g_ch_d, 136, 240), (1, g_ch_recon, 1, 1)]),
}


def _make(factory):
    return factory()


def export(net, shapes, path):
    net.eval()
    args = tuple(torch.randn(*s) for s in shapes)
    torch.onnx.export(net, args, path, dynamo=False, opset_version=OPSET,
                      input_names=[f"in{i}" for i in range(len(args))],
                      output_names=["out0"])
    m = onnx.load(path)
    ops = Counter(n.op_type for n in m.graph.node)
    n_ct = ops.get("ConvTranspose", 0); n_rs = ops.get("Resize", 0); n_dts = ops.get("DepthToSpace", 0)
    print(f"  {os.path.basename(path):32s} ConvTranspose={n_ct} Resize={n_rs} DepthToSpace={n_dts}")
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
    print(f"DCVC-RK {label} export -> {args.out_dir}  (opset {OPSET}, ffn_expansion={args.ffn_expansion})")
    shapes_manifest = {}
    for name in (args.subnets or list(registry)):
        if name not in registry:
            print(f"  skip unknown: {name}"); continue
        factory, shapes = registry[name]
        export(_make(factory), shapes, os.path.join(args.out_dir, name + ".onnx"))
        shapes_manifest[name] = [list(s) for s in shapes]
    # Single source of truth for input shapes: consumed verbatim by build_rknn.py.
    # Changing a channel width (e.g. g_ch_enc_dec) re-derives every shape here, so the
    # build stage can never bake a stale/old width into the RKNN graph.
    with open(os.path.join(args.out_dir, "shapes.json"), "w") as f:
        json.dump({"opset": OPSET, "subnets": shapes_manifest}, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()
