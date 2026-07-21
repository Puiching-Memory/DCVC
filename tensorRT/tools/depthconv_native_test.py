#!/usr/bin/env python3
"""DepthConv parity using TRT native add_plugin_v3 (no ONNX/graphsurgeon)."""
import ctypes, os, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src/cpp"))

ctypes.CDLL(str(ROOT/"tensorRT/build/plugin_demo/libdcvc_depthconv.so"), mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt
from src.layers.layers import DepthConvBlock

def build_and_run(name, in_ch, out_ch, H, W, has_adaptor, shortcut):
    dev = "cuda:0"
    blk = DepthConvBlock(in_ch, out_ch, shortcut=shortcut,
                         force_adaptor=has_adaptor and in_ch==out_ch).to(dev).half().eval()
    dc, ffn = blk.dc, blk.ffn
    g = torch.Generator(dev).manual_seed(42)
    x = (torch.rand((1, in_ch, H, W), device=dev, dtype=torch.float16, generator=g) * 2 - 1)
    with torch.no_grad():
        ref = blk(x).cpu().numpy()

    # Extract weights as float32 numpy
    def wn(t): return t.data.cpu().numpy().astype(np.float32).flatten()
    def bn(t): return t.data.cpu().numpy().astype(np.float32).flatten()
    dw_w = dc[2].weight.data.view(dc[2].weight.shape[0], -1).cpu().numpy().astype(np.float32).flatten()

    # Find plugin creator
    registry = trt.get_plugin_registry()
    creator = None
    for c in registry.all_creators:
        if c.name == "DcvcDepthConv" and c.plugin_version == "1":
            creator = c; break
    if not creator:
        print(f"[ERR] {name}: DcvcDepthConv creator not found"); return False

    # Build plugin field collection
    fields = []
    def add_field(fname, data):
        fields.append(trt.PluginField(fname, data, trt.PluginFieldType.FLOAT32))
    add_field("dc0_w", wn(dc[0].weight))
    add_field("dc0_b", bn(dc[0].bias))
    add_field("dc_dw", dw_w)
    add_field("dc_db", bn(dc[2].bias))
    add_field("dc3_w", wn(dc[3].weight))
    add_field("dc3_b", bn(dc[3].bias))
    add_field("ffn0_w", wn(ffn[0].weight))
    add_field("ffn0_b", bn(ffn[0].bias))
    add_field("ffn2_w", wn(ffn[2].weight))
    add_field("ffn2_b", bn(ffn[2].bias))
    if blk.adaptor is not None:
        add_field("ad_w", wn(blk.adaptor.weight))
        add_field("ad_b", bn(blk.adaptor.bias))
    fields.append(trt.PluginField("in_ch", np.array([in_ch], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("out_ch", np.array([out_ch], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("has_adaptor", np.array([1.0 if blk.adaptor else 0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fields.append(trt.PluginField("shortcut", np.array([1.0 if shortcut else 0.0], dtype=np.float32), trt.PluginFieldType.FLOAT32))

    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin(name, fc, trt.TensorRTPhase.BUILD)

    # Build network with plugin
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x_tensor = network.add_input("x", trt.float16, [1, in_ch, H, W])
    layer = network.add_plugin_v3([x_tensor], [], plugin)
    layer.get_output(0).name = "out"
    network.mark_output(layer.get_output(0))

    print(f"  plugin output shape: {layer.get_output(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", [1, in_ch, H, W], [1, in_ch, H, W], [1, in_ch, H, W])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if not serialized:
        print(f"[FAIL] {name}: build failed"); return False

    # Run
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    context.set_input_shape("x", [1, in_ch, H, W])
    out_shape = tuple(int(s) for s in context.get_tensor_shape("out"))
    xt = x.contiguous()
    out_t = torch.empty(out_shape, dtype=torch.float16, device=dev)
    context.set_tensor_address("x", xt.data_ptr())
    context.set_tensor_address("out", out_t.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream(dev).cuda_stream)
    torch.cuda.synchronize()
    got = out_t.cpu().numpy()

    err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    rel = err / (float(np.abs(ref).max()) + 1e-6)
    ok = rel < 0.02
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs={err:.3e} rel={rel:.4f}  "
          f"ref[{ref.min():.2f},{ref.max():.2f}] got[{got.min():.2f},{got.max():.2f}]")
    return ok

allok = True
print("=== DepthConv TRT native plugin parity ===\n")
allok &= build_and_run("no-adapt C=64", 64, 64, 32, 32, False, False)
allok &= build_and_run("adapt 32→64", 32, 64, 16, 16, True, False)
allok &= build_and_run("shortcut C=48", 48, 48, 20, 20, False, True)
allok &= build_and_run("big C=368", 368, 368, 32, 32, False, False)
print(f"\nALL {'PASS' if allok else 'FAIL'}")
sys.exit(0 if allok else 1)
