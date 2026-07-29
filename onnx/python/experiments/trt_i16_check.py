#!/usr/bin/env python3
"""Numerics check: ORT int16 QDQ vs ORT fp32-emu vs TRT fp32-emu engines."""
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

REPO = os.path.dirname(os.path.abspath(__file__))
logger = trt.Logger(trt.Logger.ERROR)


def build_engine(path, shapes):
    b = trt.Builder(logger)
    net = b.create_network(0)
    p = trt.OnnxParser(net, logger)
    with open(path, "rb") as f:
        if not p.parse(f.read()):
            for i in range(p.num_errors):
                print(p.get_error(i))
            sys.exit(1)
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    prof = b.create_optimization_profile()
    for i in range(net.num_inputs):
        t = net.get_input(i)
        if -1 in t.shape:
            s = shapes[t.name]
            prof.set_shape(t.name, s, s, s)
    cfg.add_optimization_profile(prof)
    blob = b.build_serialized_network(net, cfg)
    rt = trt.Runtime(logger)
    return rt.deserialize_cuda_engine(blob)


def trt_run(engine, feed):
    ctx = engine.create_execution_context()
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            arr = np.ascontiguousarray(feed[name])
            ctx.set_input_shape(name, arr.shape)
            d = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(d, arr)
            ctx.set_tensor_address(name, int(d))
        else:
            shape = tuple(ctx.get_tensor_shape(name))
            arr = np.empty(shape, dtype=np.float32)
            d = cuda.mem_alloc(arr.nbytes)
            ctx.set_tensor_address(name, int(d))
            outs[name] = (arr, d)
    stream = cuda.Stream()
    ctx.execute_async_v3(stream.handle)
    for arr, d in outs.values():
        cuda.memcpy_dtoh(arr, d)
    stream.synchronize()
    return {k: v[0] for k, v in outs.items()}


def load_inputs(net):
    p = f"calib_trt_i8/{net}/sample_12.npz"
    if os.path.exists(p):
        z = np.load(p)
        return {k: z[k] for k in z.files}
    f = sorted(glob.glob(f"calib_data_rd/{net}/*.npy"))[-1]
    return {"in0": np.load(f)}


def main():
    nets = ["hyper_dec", "intra_synthesis"]
    shapes_map = {
        "hyper_dec": {"in0": (1, 128, 4, 4)},
        "intra_synthesis": {"in0": (1, 256, 16, 16), "in1": (1, 368, 1, 1)},
    }
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    for net in nets:
        feed = load_inputs(net)
        s16 = ort.InferenceSession(f"../models_int16/{net}.onnx", sess_options=opts,
                                   providers=["CPUExecutionProvider"])
        semu = ort.InferenceSession(f"../models_int16_trtemu/{net}.onnx", sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        o16 = s16.run(None, feed)[0]
        oemu = semu.run(None, feed)[0]
        eng = build_engine(f"../models_int16_trtemu/{net}.onnx", shapes_map[net])
        otrt = list(trt_run(eng, feed).values())[0]
        d_emu = np.abs(o16.astype(np.float64) - oemu.astype(np.float64)).max()
        d_trt = np.abs(o16.astype(np.float64) - otrt.astype(np.float64)).max()
        bitexact = np.array_equal(o16.tobytes(), oemu.tobytes())
        print(f"{net}: |int16-emu_ort|_max={d_emu:.3e} (bitexact={bitexact})  "
              f"|int16-trt|_max={d_trt:.3e}  out_range={np.abs(o16).max():.3f}")


if __name__ == "__main__":
    main()
