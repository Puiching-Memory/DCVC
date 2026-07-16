#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Build TRT engines for DCVC-RT I-frame subgraphs using fused plugins.

Manually constructs TRT networks by walking the PyTorch model, using
add_plugin_v3 for DepthConvBlock/SubpelConv2x and standard TRT layers
for everything else. No OnnxParser dependency.
"""
import ctypes, os, sys, json
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
from src.models.image_model import DMCI, g_ch_src, g_ch_enc_dec
from src.layers.layers import DepthConvBlock, SubpelConv2x, ResidualBlockUpsample, ResidualBlockWithStride2
from src.utils.common import get_state_dict

def wn(t): return t.data.cpu().numpy().astype(np.float32).flatten()
def bn(t): return t.data.cpu().numpy().astype(np.float32).flatten()

_creators = {}
def get_creator(name):
    if name not in _creators:
        for c in trt.get_plugin_registry().all_creators:
            if c.name == name and c.plugin_version == "1":
                _creators[name] = c; break
    return _creators[name]


def add_depthconv(network, input_t, block, name, dev):
    dc, ffn = block.dc, block.ffn
    fields = []
    def af(n, d): fields.append(trt.PluginField(n, d, trt.PluginFieldType.FLOAT32))
    af("dc0_w", wn(dc[0].weight)); af("dc0_b", bn(dc[0].bias))
    dw_w = dc[2].weight.data.view(dc[2].weight.shape[0], -1).cpu().numpy().astype(np.float32).flatten()
    af("dc_dw", dw_w); af("dc_db", bn(dc[2].bias))
    af("dc3_w", wn(dc[3].weight)); af("dc3_b", bn(dc[3].bias))
    af("ffn0_w", wn(ffn[0].weight)); af("ffn0_b", bn(ffn[0].bias))
    af("ffn2_w", wn(ffn[2].weight)); af("ffn2_b", bn(ffn[2].bias))
    if block.adaptor is not None:
        af("ad_w", wn(block.adaptor.weight)); af("ad_b", bn(block.adaptor.bias))
    in_ch = dc[0].in_channels; out_ch = dc[0].out_channels
    af("in_ch", np.array([in_ch], dtype=np.float32))
    af("out_ch", np.array([out_ch], dtype=np.float32))
    af("has_adaptor", np.array([1.0 if block.adaptor else 0.0], dtype=np.float32))
    af("shortcut", np.array([1.0 if block.shortcut else 0.0], dtype=np.float32))
    plugin = get_creator("DcvcDepthConv").create_plugin(name, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD)
    return network.add_plugin_v3([input_t], [], plugin).get_output(0)


def add_subpel(network, input_t, subpel, name):
    conv0 = subpel.conv[0]
    fields = [
        trt.PluginField("weight", wn(conv0.weight), trt.PluginFieldType.FLOAT32),
        trt.PluginField("bias", bn(conv0.bias), trt.PluginFieldType.FLOAT32),
        trt.PluginField("in_ch", np.array([conv0.in_channels], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("out_ch", np.array([conv0.out_channels // 4], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("kernel_size", np.array([conv0.kernel_size[0]], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("padding", np.array([subpel.padding], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("has_cat", np.array([0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("cat_at_front", np.array([1.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("cat_ch", np.array([0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32),
    ]
    plugin = get_creator("DcvcSubpelConv2x").create_plugin(name, trt.PluginFieldCollection(fields), trt.TensorRTPhase.BUILD)
    return network.add_plugin_v3([input_t], [], plugin).get_output(0)


def add_conv2d(network, input_t, conv, name):
    w = conv.weight.data.cpu().numpy().astype(np.float16)
    b = conv.bias.data.cpu().numpy().astype(np.float16) if conv.bias is not None else None
    layer = network.add_convolution_nd(input_t, conv.out_channels,
        trt.DimsHW(conv.kernel_size[0], conv.kernel_size[1]), w, b)
    layer.stride_nd = trt.DimsHW(conv.stride[0], conv.stride[1])
    layer.padding_nd = trt.DimsHW(conv.padding[0], conv.padding[1])
    layer.name = name
    return layer.get_output(0)


def add_pixel_unshuffle(network, input_t, r, C, H, W):
    s1 = network.add_shuffle(input_t)
    s1.reshape_dims = trt.Dims([1, C, H // r, r, W // r, r])
    s1.second_transpose = trt.Permutation([0, 1, 3, 5, 2, 4])
    s2 = network.add_shuffle(s1.get_output(0))
    s2.reshape_dims = trt.Dims([1, C * r * r, H // r, W // r])
    return s2.get_output(0)


def add_pixel_shuffle(network, input_t, r, C, H, W):
    s1 = network.add_shuffle(input_t)
    s1.reshape_dims = trt.Dims([1, C // (r * r), r, r, H, W])
    s1.second_transpose = trt.Permutation([0, 1, 4, 2, 5, 3])
    s2 = network.add_shuffle(s1.get_output(0))
    s2.reshape_dims = trt.Dims([1, C // (r * r), H * r, W * r])
    return s2.get_output(0)


def build_and_run(name, build_fn, input_specs, ref_fn, dev="cuda:0", ws_gb=2):
    print(f"\n=== {name} ===")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)

    input_tensors = {}
    for iname, (shape, _) in input_specs.items():
        input_tensors[iname] = network.add_input(iname, trt.float16, list(shape))

    out_t = build_fn(network, input_tensors)
    network.mark_output(out_t)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gb * (1 << 30))
    profile = builder.create_optimization_profile()
    for iname, (shape, _) in input_specs.items():
        profile.set_shape(iname, list(shape), list(shape), list(shape))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if not serialized:
        print(f"  [FAIL] build failed"); return False

    engine_dir = ROOT / "native/assets/engines"
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / f"{name}.engine").write_bytes(serialized)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    inputs = {}
    for iname, (shape, gen) in input_specs.items():
        inputs[iname] = gen()
        context.set_input_shape(iname, list(shape))

    out_name = engine.get_tensor_name(engine.num_io_tensors - 1)
    out_shape = tuple(int(s) for s in context.get_tensor_shape(out_name))
    out_buf = torch.empty(out_shape, dtype=torch.float16, device=dev)

    stream = torch.cuda.Stream(dev)
    for iname, tensor in inputs.items():
        context.set_tensor_address(iname, tensor.contiguous().data_ptr())
    context.set_tensor_address(out_name, out_buf.data_ptr())
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    ref = ref_fn(inputs)
    err = float((out_buf.float() - ref.float()).abs().max())
    rel = err / (float(ref.float().abs().max()) + 1e-6)
    ok = rel < 0.05
    print(f"  [{'PASS' if ok else 'FAIL'}] max_abs={err:.3e} rel={rel:.4f}")
    return ok


def main():
    dev = "cuda:0"
    net = DMCI().to(dev).half().eval()
    net.load_state_dict(get_state_dict(str(ROOT/"checkpoints/cvpr2025_image.pth.tar")))
    H, W, N = 256, 256, 256

    g = torch.Generator(dev).manual_seed(7)

    # === IntraEncoder ===
    def build_enc(network, inp):
        x = inp["x"]; q = inp["q"]
        t = add_pixel_unshuffle(network, x, 8, 3, H, W)
        t = add_depthconv(network, t, net.enc.enc_1, "enc_1", dev)
        t = network.add_elementwise(t, q, trt.ElementWiseOperation.PROD).get_output(0)
        for i, mod in enumerate(net.enc.enc_2):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"enc_2.{i}", dev)
            elif isinstance(mod, nn.Conv2d):
                t = add_conv2d(network, t, mod, f"enc_2.{i}")
        return t

    x_gen = lambda: (torch.rand((1, 3, H, W), device=dev, dtype=torch.float16, generator=g) * 0.8 + 0.1)
    q_gen = lambda: torch.ones((1, g_ch_enc_dec, 1, 1), device=dev, dtype=torch.float16)
    ref_enc = lambda inp: net.enc(inp["x"], inp["q"])
    build_and_run("intra_analysis", build_enc,
                  {"x": ((1, 3, H, W), x_gen), "q": ((1, g_ch_enc_dec, 1, 1), q_gen)},
                  ref_enc, dev)

    # === HyperEnc ===
    def build_hyperenc(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.hyper_enc):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"he.{i}", dev)
            elif isinstance(mod, ResidualBlockWithStride2):
                t = add_conv2d(network, t, mod.down, f"he.{i}.down")
                t = add_depthconv(network, t, mod.conv, f"he.{i}.conv", dev)
        return t

    y_gen = lambda: torch.randn((1, N, H//16, W//16), device=dev, dtype=torch.float16, generator=g)
    ref_he = lambda inp: net.hyper_enc(inp["in0"])
    build_and_run("intra_hyper_enc", build_hyperenc,
                  {"in0": ((1, N, H//16, W//16), y_gen)}, ref_he, dev)

    # === IntraDecoder ===
    def build_dec(network, inp):
        y = inp["in0"]; q = inp["in1"]
        t = y
        for i, mod in enumerate(net.dec.dec_1):
            if isinstance(mod, ResidualBlockUpsample):
                t = add_subpel(network, t, mod.up, f"dec_1.{i}.up")
                t = add_depthconv(network, t, mod.conv, f"dec_1.{i}.conv", dev)
            elif isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"dec_1.{i}", dev)
        t = network.add_elementwise(t, q, trt.ElementWiseOperation.PROD).get_output(0)
        t = add_depthconv(network, t, net.dec.dec_2, "dec_2", dev)
        t = add_pixel_shuffle(network, t, 8, g_ch_src, H//8, W//8)
        return t

    yhat_gen = lambda: torch.randn((1, N, H//16, W//16), device=dev, dtype=torch.float16, generator=g)
    qd_gen = lambda: torch.ones((1, g_ch_enc_dec, 1, 1), device=dev, dtype=torch.float16)
    ref_dec = lambda inp: net.dec(inp["in0"], inp["in1"])
    build_and_run("intra_synthesis", build_dec,
                  {"in0": ((1, N, H//16, W//16), yhat_gen), "in1": ((1, g_ch_enc_dec, 1, 1), qd_gen)},
                  ref_dec, dev)

    # === HyperDec ===
    zH, zW, ZC = 4, 4, net.z_channel
    def build_hyperdec(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.hyper_dec):
            if isinstance(mod, ResidualBlockUpsample):
                t = add_subpel(network, t, mod.up, f"hd.{i}.up")
                t = add_depthconv(network, t, mod.conv, f"hd.{i}.conv", dev)
            elif isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"hd.{i}", dev)
        return t

    z_gen = lambda: torch.randn((1, ZC, zH, zW), device=dev, dtype=torch.float16, generator=g)
    ref_hd = lambda inp: net.hyper_dec(inp["in0"])
    build_and_run("hyper_dec", build_hyperdec,
                  {"in0": ((1, ZC, zH, zW), z_gen)}, ref_hd, dev)

    # === YPriorFusion ===
    def build_ypf(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.y_prior_fusion):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"ypf.{i}", dev)
            elif isinstance(mod, nn.Conv2d):
                t = add_conv2d(network, t, mod, f"ypf.{i}")
        return t

    pf_gen = lambda: torch.randn((1, N, 16, 16), device=dev, dtype=torch.float16, generator=g)
    ref_ypf = lambda inp: net.y_prior_fusion(inp["in0"])
    build_and_run("y_prior_fusion", build_ypf,
                  {"in0": ((1, N, 16, 16), pf_gen)}, ref_ypf, dev)

    # === YSpatialPriorReduction ===
    def build_yspr(network, inp):
        return add_conv2d(network, inp["in0"], net.y_spatial_prior_reduction, "yspr")

    yspr_gen = lambda: torch.randn((1, N * 2 + 2, 16, 16), device=dev, dtype=torch.float16, generator=g)
    ref_yspr = lambda inp: net.y_spatial_prior_reduction(inp["in0"])
    build_and_run("y_spatial_prior_reduction", build_yspr,
                  {"in0": ((1, N * 2 + 2, 16, 16), yspr_gen)}, ref_yspr, dev)

    # === YSpatialPriorAdaptors (3 separate engines, different weights) ===
    for adapt_idx in (1, 2, 3):
        adapt_mod = getattr(net, f"y_spatial_prior_adaptor_{adapt_idx}")
        def build_yspa(network, inp, am=adapt_mod, idx=adapt_idx):
            return add_depthconv(network, inp["in0"], am, f"yspa{idx}", dev)
        yspa_gen = lambda: torch.randn((1, N * 2, 16, 16), device=dev, dtype=torch.float16, generator=g)
        ref_yspa = lambda inp, am=adapt_mod: am(inp["in0"])
        build_and_run(f"y_spatial_prior_adaptor_{adapt_idx}", build_yspa,
                      {"in0": ((1, N * 2, 16, 16), yspa_gen)}, ref_yspa, dev)

    # === YSpatialPrior ===
    def build_ysp(network, inp):
        t = inp["in0"]
        for i, mod in enumerate(net.y_spatial_prior):
            if isinstance(mod, DepthConvBlock):
                t = add_depthconv(network, t, mod, f"ysp.{i}", dev)
            elif isinstance(mod, nn.Conv2d):
                t = add_conv2d(network, t, mod, f"ysp.{i}")
        return t

    ysp_gen = lambda: torch.randn((1, N * 2, 16, 16), device=dev, dtype=torch.float16, generator=g)
    ref_ysp = lambda inp: net.y_spatial_prior(inp["in0"])
    build_and_run("y_spatial_prior", build_ysp,
                  {"in0": ((1, N * 2, 16, 16), ysp_gen)}, ref_ysp, dev)

if __name__ == "__main__":
    main()
