#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Closed-loop demo: ONNX custom op node -> TRT plugin -> CUDA kernel -> parity.

Proves the custom-operator pipeline end-to-end with a single op
(process_with_mask -> y_hat output) before porting the rest of the kernels.
"""
from __future__ import annotations
import ctypes
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 1) Load plugin .so -> static registrar adds the creator to the global registry.
PLUGIN_SO = ROOT / "tensorRT/build/plugin_demo/libdcvc_process_mask.so"
ctypes.CDLL(str(PLUGIN_SO), mode=ctypes.RTLD_GLOBAL)
print("[demo] loaded plugin:", PLUGIN_SO.name)

# 2) Build a minimal ONNX graph with the custom op node via onnx-graphsurgeon.
import onnx_graphsurgeon as gs
import onnx
from onnx import TensorProto, helper

C, H, W = 128, 16, 16
DT = TensorProto.FLOAT16

yi = gs.Variable("y", DT, [1, C, H, W])
si = gs.Variable("scales", DT, [1, C, H, W])
mi = gs.Variable("means", DT, [1, C, H, W])
ki = gs.Variable("mask", DT, [1, C, H, W])
yo = gs.Variable("y_hat", DT, [1, C, H, W])

node = gs.Node(op="DcvcProcessWithMask", name="pm0",
               attrs={"force_zero_thres": np.float32(-1.0)},
               inputs=[yi, si, mi, ki], outputs=[yo])
graph = gs.Graph(nodes=[node], inputs=[yi, si, mi, ki], outputs=[yo],
                 opset=17)
onnx_model = gs.export_onnx(graph)
onnx_path = ROOT / "tensorRT/build/plugin_demo/process_mask.onnx"
onnx.save(onnx_model, str(onnx_path))
print("[demo] wrote ONNX:", onnx_path.name, "| op_type:", node.op)

# 3) Build the TRT engine (OnnxParser resolves the custom node via the registry).
import tensorrt as trt
trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, "")  # built-in plugins
builder = trt.Builder(trt_logger)
network = builder.create_network(0)  # explicit batch is default in TRT 11
parser = trt.OnnxParser(network, trt_logger)
if not parser.parse(onnx_model.SerializeToString()):
    for i in range(parser.num_errors):
        print("PARSE ERR:", parser.get_error(i))
    sys.exit(1)
print("[demo] OnnxParser resolved custom node OK")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
profile = builder.create_optimization_profile()
for tname in ["y", "scales", "means", "mask"]:
    profile.set_shape(tname, [1, C, H, W], [1, C, H, W], [1, C, H, W])
config.add_optimization_profile(profile)
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    print("[demo] engine build FAILED")
    sys.exit(1)
print("[demo] engine built:", serialized.nbytes, "bytes")

# 4) Run the engine on deterministic FP16 inputs.
runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(serialized)
context = engine.create_execution_context()
context.set_input_shape("y", [1, C, H, W])

rng = np.random.default_rng(42)
def rand16(): return rng.standard_normal([1, C, H, W]).astype(np.float16)
y_d, s_d, m_d, k_d = rand16(), np.abs(rand16()), rand16(), (rng.random([1, C, H, W]) > 0.2).astype(np.float16)

import torch
def to_trt(arr, dev="cuda:0"):
    t = torch.from_numpy(arr).to(dev)
    return t

dev = "cuda:0"
yt = to_trt(y_d, dev); st = to_trt(s_d, dev); mt = to_trt(m_d, dev); kt = to_trt(k_d, dev)
out_t = torch.empty([1, C, H, W], dtype=torch.float16, device=dev)
bindings = [yt.data_ptr(), st.data_ptr(), mt.data_ptr(), kt.data_ptr(), out_t.data_ptr()]
for i, name in enumerate(["y", "scales", "means", "mask", "y_hat"]):
    try:
        context.set_tensor_address(name, bindings[i])
    except Exception:
        pass
stream = torch.cuda.current_stream(dev).cuda_stream
context.execute_async_v3(stream)
torch.cuda.synchronize()
got = out_t.cpu().numpy()

# 5) PyTorch reference (fallback semantics) in FP32 to match plugin kernel precision.
yf, sf, mf, kf = yt.float(), st.float(), mt.float(), kt.float()
means_hat = (mf * kf)
y_res = (yf - means_hat) * kf
y_q = torch.round(y_res)
y_q = torch.clamp(y_q, -128., 127.)
ref = (y_q + means_hat).cpu().numpy().astype(np.float16)

max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
ok = max_err < 1e-3
print(f"\n[{'PASS' if ok else 'FAIL'}] TRT plugin vs PyTorch: max_abs_err={max_err:.3e}")
print(f"  got range [{got.min():.3f}, {got.max():.3f}]  ref range [{ref.min():.3f}, {ref.max():.3f}]")
sys.exit(0 if ok else 1)
