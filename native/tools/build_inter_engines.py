#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Build TRT engines for DCVC-RT P-frame (inter) subgraphs.

Exports all engines needed for the inter pipeline:
  feature_adaptor_i, feature_adaptor_p, feature_extractor_p1/p2,
  inter_encoder, inter_hyper_enc, inter_hyper_dec, temporal_prior,
  inter_prior_fusion, inter_spatial_prior, inter_decoder, recon_generation

Reuses helpers from build_plugin_engines.py.
"""
import ctypes, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src/cpp"))
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")

for so in ["libdcvc_depthconv.so", "libdcvc_subpel.so"]:
    ctypes.CDLL(str(ROOT/"native/build/plugin_demo"/so), mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt
from src.models.video_model import (DMC, g_ch_src_d, g_ch_recon, g_ch_y, g_ch_z,
                                     g_ch_d, qp_shift, extra_qp)
from src.layers.layers import (DepthConvBlock, SubpelConv2x,
                                ResidualBlockUpsample, ResidualBlockWithStride2)
from src.utils.common import get_state_dict

# Import helpers from the intra script
from build_plugin_engines import (wn, bn, get_creator, add_depthconv, add_subpel,
                                   add_conv2d, add_pixel_unshuffle, add_pixel_shuffle,
                                   build_and_run)

def add_subpel_cat(network, input_t, subpel, name, cat_ch, cat_at_front=False):
    """SubpelConv2x with concatenation (for inter Decoder)."""
    conv0 = subpel.conv[0]
    fields = [
        trt.PluginField("weight", wn(conv0.weight), trt.PluginFieldType.FLOAT32),
        trt.PluginField("bias", bn(conv0.bias), trt.PluginFieldType.FLOAT32),
        trt.PluginField("in_ch", np.array([conv0.in_channels], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("out_ch", np.array([conv0.out_channels // 4], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("kernel_size", np.array([conv0.kernel_size[0]], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("padding", np.array([subpel.padding], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("has_cat", np.array([1.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("cat_at_front", np.array([1.0 if cat_at_front else 0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("cat_ch", np.array([float(cat_ch)], dtype=np.float32), trt.PluginFieldType.FLOAT32),
    ]
    plugin = get_creator("DcvcSubpelConv2x").create_plugin(name, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD)
    return network.add_plugin_v3([input_t], [], plugin).get_output(0)

def add_residual_stride2(network, input_t, block, name, dev):
    """ResidualBlockWithStride2: Conv2d(stride2) → DepthConvBlock(shortcut)."""
    t = add_conv2d(network, input_t, block.down, f"{name}.down")
    t = add_depthconv(network, t, block.conv, f"{name}.conv", dev)
    return t

def add_residual_upsample(network, input_t, block, name, dev):
    """ResidualBlockUpsample: SubpelConv2x → DepthConvBlock(shortcut)."""
    t = add_subpel(network, input_t, block.up, f"{name}.up")
    t = add_depthconv(network, t, block.conv, f"{name}.conv", dev)
    return t


def main():
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(42)
    net = DMC().to(dev).half().eval()
    net.load_state_dict(get_state_dict(str(ROOT/"checkpoints/cvpr2025_video.pth.tar")))
    net.update()

    H, W = 256, 256  # standard test resolution
    yH, yW = H // 16, W // 16   # y latent spatial size
    zH, zW = H // 64, W // 64   # z latent spatial size
    featH, featW = H // 8, W // 8  # feature spatial size
    passed, failed = 0, 0

    def run(name, build_fn, specs, ref_fn):
        nonlocal passed, failed
        if build_and_run(name, build_fn, specs, ref_fn, dev):
            passed += 1
        else:
            failed += 1

    print(f"=== Building inter engines (H={H} W={W}, yH={yH} yW={yW}) ===")
    print(f"  g_ch_y={g_ch_y} g_ch_z={g_ch_z} g_ch_d={g_ch_d} g_ch_recon={g_ch_recon}")

    # ---- 1. feature_adaptor_i: DepthConvBlock(192, 256) ----
    # Input: pixel_unshuffled ref frame [1, 192, H/8, W/8]
    def build_fai(network, inp):
        return add_depthconv(network, inp["in0"], net.feature_adaptor_i, "fai", dev)
    run("inter_feature_adaptor_i", build_fai,
        {"in0": ((1, g_ch_src_d, featH, featW),
                 lambda: torch.rand((1, g_ch_src_d, featH, featW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.feature_adaptor_i(inp["in0"]))

    # ---- 2. feature_adaptor_p: Conv2d(256, 256, 1) ----
    def build_fap(network, inp):
        return add_conv2d(network, inp["in0"], net.feature_adaptor_p, "fap")
    run("inter_feature_adaptor_p", build_fap,
        {"in0": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.feature_adaptor_p(inp["in0"]))

    # ---- 3. feature_extractor part1: conv1 → mul q_feature ----
    # conv1 = 2 DepthConvBlocks, output * q_feature
    def build_fe1(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.feature_extractor.conv1):
            t = add_depthconv(network, t, mod, f"fe1.conv1.{i}", dev)
        t = network.add_elementwise(t, inp["q"], trt.ElementWiseOperation.PROD).get_output(0)
        return t
    run("inter_feature_extractor_p1", build_fe1,
        {"in0": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g)),
         "q": ((1, g_ch_d, 1, 1),
               lambda: torch.ones((1, g_ch_d, 1, 1), device=dev, dtype=torch.float16))},
        lambda inp: net.feature_extractor.forward_part1(inp["in0"], inp["q"])[0])

    # ---- 4. feature_extractor part2: conv2 (4 DepthConvBlocks) ----
    def build_fe2(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.feature_extractor.conv2):
            t = add_depthconv(network, t, mod, f"fe2.conv2.{i}", dev)
        return t
    run("inter_feature_extractor_p2", build_fe2,
        {"in0": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.feature_extractor.forward_part2(inp["in0"]))

    # ---- 5. inter_encoder: conv1 → cat(ctx) → conv2 → conv3 → mul q → down ----
    # Inputs: x [1,3,H,W], ctx [1,256,H/8,W/8], q [1,256,1,1]
    def build_enc(network, inp):
        x = inp["x"]; ctx = inp["ctx"]; q = inp["q"]
        t = add_pixel_unshuffle(network, x, 8, 3, H, W)  # [1,192,H/8,W/8]
        t = add_conv2d(network, t, net.encoder.conv1, "enc.conv1")  # [1,256,H/8,W/8]
        cat = network.add_concatenation([t, ctx]); cat.axis = 1; t = cat.get_output(0)  # [1,512,H/8,W/8]
        for i, mod in enumerate(net.encoder.conv2):
            t = add_depthconv(network, t, mod, f"enc.conv2.{i}", dev)
        t = add_depthconv(network, t, net.encoder.conv3, "enc.conv3", dev)
        t = network.add_elementwise(t, q, trt.ElementWiseOperation.PROD).get_output(0)
        t = add_conv2d(network, t, net.encoder.down, "enc.down")
        return t
    run("inter_encoder", build_enc,
        {"x": ((1, 3, H, W), lambda: torch.rand((1,3,H,W), device=dev, dtype=torch.float16, generator=g)),
         "ctx": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g)),
         "q": ((1, g_ch_d, 1, 1),
               lambda: torch.ones((1, g_ch_d, 1, 1), device=dev, dtype=torch.float16))},
        lambda inp: net.encoder.forward_torch(
            torch.nn.functional.pixel_unshuffle(inp["x"], 8), inp["ctx"], inp["q"]))

    # ---- 6. inter_hyper_enc: DepthConvBlock + 2×ResidualBlockWithStride2 ----
    def build_he(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.hyper_encoder.conv):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"he.{i}", dev)
            elif isinstance(mod, ResidualBlockWithStride2):
                t = add_residual_stride2(network, t, mod, f"he.{i}", dev)
        return t
    run("inter_hyper_enc", build_he,
        {"in0": ((1, g_ch_y, yH, yW),
                 lambda: torch.randn((1, g_ch_y, yH, yW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.hyper_encoder(inp["in0"]))

    # ---- 7. inter_hyper_dec: 2×ResidualBlockUpsample + DepthConvBlock ----
    def build_hd(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.hyper_decoder.conv):
            if isinstance(mod, ResidualBlockUpsample):
                t = add_residual_upsample(network, t, mod, f"hd.{i}", dev)
            elif isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"hd.{i}", dev)
        return t
    run("inter_hyper_dec", build_hd,
        {"in0": ((1, g_ch_z, zH, zW),
                 lambda: torch.randn((1, g_ch_z, zH, zW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.hyper_decoder(inp["in0"]))

    # ---- 8. temporal_prior_encoder: ResidualBlockWithStride2(256, 256) ----
    def build_tp(network, inp):
        return add_residual_stride2(network, inp["in0"], net.temporal_prior_encoder, "tp", dev)
    run("inter_temporal_prior", build_tp,
        {"in0": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.temporal_prior_encoder(inp["in0"]))

    # ---- 9. inter_prior_fusion: PriorFusion (3×128=384 → 3×128=384) ----
    def build_pf(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.y_prior_fusion.conv):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"pf.{i}", dev)
            elif isinstance(mod, nn.Conv2d):
                t = add_conv2d(network, t, mod, f"pf.{i}")
        return t
    run("inter_prior_fusion", build_pf,
        {"in0": ((1, g_ch_y * 3, yH, yW),
                 lambda: torch.randn((1, g_ch_y * 3, yH, yW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.y_prior_fusion(inp["in0"]))

    # ---- 10. inter_spatial_prior: SpatialPrior (4×128=512 → 2×128=256) ----
    def build_sp(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.y_spatial_prior.conv):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"sp.{i}", dev)
            elif isinstance(mod, nn.Conv2d):
                t = add_conv2d(network, t, mod, f"sp.{i}")
        return t
    run("inter_spatial_prior", build_sp,
        {"in0": ((1, g_ch_y * 4, yH, yW),
                 lambda: torch.randn((1, g_ch_y * 4, yH, yW), device=dev, dtype=torch.float16, generator=g))},
        lambda inp: net.y_spatial_prior(inp["in0"]))

    # ---- 11. inter_decoder: up(x) → cat(ctx) → conv1 → conv2 → mul q ----
    # Inputs: y_hat [1,128,yH,yW], ctx [1,256,featH,featW], q [1,256,1,1]
    def build_dec(network, inp):
        x = inp["in0"]; ctx = inp["ctx"]; q = inp["q"]
        # SubpelConv2x without cat, then TRT concatenation
        t = add_subpel(network, x, net.decoder.up, "dec.up")  # [1,256,featH,featW]
        cat = network.add_concatenation([t, ctx]); cat.axis = 1
        t = cat.get_output(0)  # [1,512,featH,featW]
        for i, mod in enumerate(net.decoder.conv1):
            t = add_depthconv(network, t, mod, f"dec.conv1.{i}", dev)
        t = add_conv2d(network, t, net.decoder.conv2, "dec.conv2")
        t = network.add_elementwise(t, q, trt.ElementWiseOperation.PROD).get_output(0)
        return t
    run("inter_decoder", build_dec,
        {"in0": ((1, g_ch_y, yH, yW),
                 lambda: torch.randn((1, g_ch_y, yH, yW), device=dev, dtype=torch.float16, generator=g)),
         "ctx": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g)),
         "q": ((1, g_ch_d, 1, 1),
               lambda: torch.ones((1, g_ch_d, 1, 1), device=dev, dtype=torch.float16))},
        lambda inp: net.decoder.forward_torch(inp["in0"], inp["ctx"], inp["q"]))

    # ---- 12. recon_generation: conv → mul q → head → pixel_shuffle_8 → clamp ----
    # Inputs: feature [1,256,featH,featW], q_recon [1,320,1,1]
    def build_rg(network, inp):
        x = inp["in0"]; q = inp["q"]
        t = x
        for i, mod in enumerate(net.recon_generation_net.conv):
            t = add_depthconv(network, t, mod, f"rg.conv.{i}", dev)
        t = network.add_elementwise(t, q, trt.ElementWiseOperation.PROD).get_output(0)
        t = add_conv2d(network, t, net.recon_generation_net.head, "rg.head")
        t = add_pixel_shuffle(network, t, 8, g_ch_src_d, featH, featW)
        # clamp [0, 1] — PyTorch forward_torch applies torch.clamp(out, 0., 1.)
        cl = network.add_activation(t, trt.ActivationType.CLIP)
        cl.alpha = 0.0; cl.beta = 1.0
        t = cl.get_output(0)
        return t
    run("recon_generation", build_rg,
        {"in0": ((1, g_ch_d, featH, featW),
                 lambda: torch.randn((1, g_ch_d, featH, featW), device=dev, dtype=torch.float16, generator=g)),
         "q": ((1, g_ch_recon, 1, 1),
               lambda: torch.ones((1, g_ch_recon, 1, 1), device=dev, dtype=torch.float16))},
        lambda inp: net.recon_generation_net.forward_torch(inp["in0"], inp["q"]))

    print(f"\n=== DONE: {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
