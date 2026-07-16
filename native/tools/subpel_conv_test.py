#!/usr/bin/env python3
"""SubpelConv2x parity using TRT native add_plugin_v3."""
import ctypes, os, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"src/cpp"))

ctypes.CDLL(str(ROOT/"native/build/plugin_demo/libdcvc_subpel.so"), mode=ctypes.RTLD_GLOBAL)

import tensorrt as trt
from src.layers.layers import SubpelConv2x

def build_and_run(name, in_ch, out_ch, kernel_size, padding, H, W, has_cat=False, cat_ch=0, cat_at_front=True):
    dev = "cuda:0"
    blk = SubpelConv2x(in_ch, out_ch, kernel_size, padding=padding).to(dev).half().eval()
    g = torch.Generator(dev).manual_seed(42)
    x = (torch.rand((1, in_ch, H, W), device=dev, dtype=torch.float16, generator=g) * 2 - 1)

    to_cat = None
    if has_cat:
        to_cat = (torch.rand((1, cat_ch, H*2, W*2), device=dev, dtype=torch.float16, generator=g) * 2 - 1)

    with torch.no_grad():
        ref = blk(x, to_cat=to_cat, cat_at_front=cat_at_front).cpu().numpy()

    def wn(t): return t.data.cpu().numpy().astype(np.float32).flatten()
    def bn(t): return t.data.cpu().numpy().astype(np.float32).flatten()

    registry = trt.get_plugin_registry()
    creator = None
    for c in registry.all_creators:
        if c.name == "DcvcSubpelConv2x" and c.plugin_version == "1":
            creator = c; break
    if not creator:
        print(f"[ERR] {name}: creator not found"); return False

    fields = []
    def add_field(fname, data):
        fields.append(trt.PluginField(fname, data, trt.PluginFieldType.FLOAT32))
    add_field("weight", wn(blk.conv[0].weight))
    add_field("bias", bn(blk.conv[0].bias))
    add_field("in_ch", np.array([in_ch], dtype=np.float32))
    add_field("out_ch", np.array([out_ch], dtype=np.float32))
    add_field("kernel_size", np.array([kernel_size], dtype=np.float32))
    add_field("padding", np.array([padding], dtype=np.float32))
    add_field("has_cat", np.array([1.0 if has_cat else 0.0], dtype=np.float32))
    add_field("cat_at_front", np.array([1.0 if cat_at_front else 0.0], dtype=np.float32))
    add_field("cat_ch", np.array([cat_ch], dtype=np.float32))

    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin(name, fc, trt.TensorRTPhase.BUILD)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    x_tensor = network.add_input("x", trt.float16, [1, in_ch, H, W])
    inputs_list = [x_tensor]
    if has_cat:
        cat_tensor = network.add_input("to_cat", trt.float16, [1, cat_ch, H*2, W*2])
        inputs_list.append(cat_tensor)

    layer = network.add_plugin_v3(inputs_list, [], plugin)
    layer.get_output(0).name = "out"
    network.mark_output(layer.get_output(0))

    print(f"  plugin output shape: {layer.get_output(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", [1, in_ch, H, W], [1, in_ch, H, W], [1, in_ch, H, W])
    if has_cat:
        profile.set_shape("to_cat", [1, cat_ch, H*2, W*2], [1, cat_ch, H*2, W*2], [1, cat_ch, H*2, W*2])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if not serialized:
        print(f"[FAIL] {name}: build failed"); return False

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    context.set_input_shape("x", [1, in_ch, H, W])
    if has_cat:
        context.set_input_shape("to_cat", [1, cat_ch, H*2, W*2])
    out_shape = tuple(int(s) for s in context.get_tensor_shape("out"))
    xt = x.contiguous()
    out_t = torch.empty(out_shape, dtype=torch.float16, device=dev)
    context.set_tensor_address("x", xt.data_ptr())
    if has_cat:
        context.set_tensor_address("to_cat", to_cat.contiguous().data_ptr())
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
print("=== SubpelConv2x TRT native plugin parity ===\n")
allok &= build_and_run("k1 256->368", 256, 368, 1, 0, 16, 16)
allok &= build_and_run("k1 128->128", 128, 128, 1, 0, 16, 16)
allok &= build_and_run("k3 128->64", 128, 64, 3, 1, 16, 16)
allok &= build_and_run("k1 64->128 cat_front", 64, 128, 1, 0, 8, 8, has_cat=True, cat_ch=32, cat_at_front=True)
allok &= build_and_run("k1 64->128 cat_back", 64, 128, 1, 0, 8, 8, has_cat=True, cat_ch=32, cat_at_front=False)
print(f"\nALL {'PASS' if allok else 'FAIL'}")
sys.exit(0 if allok else 1)
