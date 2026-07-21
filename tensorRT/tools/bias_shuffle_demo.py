#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Closed-loop: DcvcBiasPixelShuffle8 TRT plugin (engine-side synthesis endpoint)."""
from __future__ import annotations
import ctypes, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ctypes.CDLL(str(ROOT / "tensorRT/build/plugin_demo/libdcvc_bias_shuffle.so"), mode=ctypes.RTLD_GLOBAL)
print("[demo] loaded bias_shuffle plugin")

import onnx_graphsurgeon as gs
import onnx
from onnx import TensorProto
import tensorrt as trt
import torch

C, H, W = 192, 8, 8   # 192 = 3*64, pixel_shuffle(8) → 3 channels at 8x
DT = TensorProto.FLOAT16

xi = gs.Variable("x", DT, [1, C, H, W])
bi = gs.Variable("bias", DT, [C])
yo = gs.Variable("out", DT, [1, C // 64, H * 8, W * 8])
node = gs.Node(op="DcvcBiasPixelShuffle8", name="bs0",
               attrs={"clamp": np.int32(1)}, inputs=[xi, bi], outputs=[yo])
graph = gs.Graph(nodes=[node], inputs=[xi, bi], outputs=[yo], opset=17)
onnx_model = gs.export_onnx(graph)
print("[demo] ONNX node:", node.op, "| out:", yo.shape)

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse(onnx_model.SerializeToString()):
    for i in range(parser.num_errors): print("ERR:", parser.get_error(i))
    sys.exit(1)
print("[demo] OnnxParser resolved custom node OK")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
profile = builder.create_optimization_profile()
profile.set_shape("x", [1, C, H, W], [1, C, H, W], [1, C, H, W])
config.add_optimization_profile(profile)
serialized = builder.build_serialized_network(network, config)
if not serialized:
    print("[demo] build FAILED"); sys.exit(1)
print(f"[demo] engine built: {serialized.nbytes} bytes")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
context = engine.create_execution_context()
context.set_input_shape("x", [1, C, H, W])

dev = "cuda:0"
rng = np.random.default_rng(11)
x_np = (rng.standard_normal([1, C, H, W]).astype(np.float16))
b_np = (rng.standard_normal([C]).astype(np.float16))
xt = torch.from_numpy(x_np).to(dev)
bt = torch.from_numpy(b_np).to(dev)
out_t = torch.empty([1, 3, H*8, W*8], dtype=torch.float16, device=dev)

for name, t in [("x", xt), ("bias", bt), ("out", out_t)]:
    try: context.set_tensor_address(name, t.data_ptr())
    except Exception: pass
context.execute_async_v3(torch.cuda.current_stream(dev).cuda_stream)
torch.cuda.synchronize()
got = out_t.cpu().numpy()

import torch.nn.functional as F
ref = (xt.float() + bt.float()[None,:,None,None]).clamp(0,1)
ref = F.pixel_shuffle(ref, 8)
max_err = float(np.abs(got.astype(np.float32) - ref.cpu().numpy().astype(np.float32)).max())
ok = max_err < 5e-3
print(f"\n[{'PASS' if ok else 'FAIL'}] DcvcBiasPixelShuffle8: max_abs_err={max_err:.3e}")
print(f"  got shape {got.shape}  ref shape {tuple(ref.shape)}")
sys.exit(0 if ok else 1)
