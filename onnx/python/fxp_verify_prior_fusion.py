#!/usr/bin/env python3
"""Compare FP32 last-Conv vs the FxpConv1x1 numeric recipe on y_prior_fusion.

Does not need the C custom op — mirrors fxp_conv1x1_f32 in NumPy and reports
output / scale-index agreement after replacing only the final 1x1.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))

LOG_SCALE_MIN = np.float32(np.log(np.float32(0.11)))
LOG_STEP_RECIP = np.float32(1.0) / (
    (np.float32(np.log(np.float32(16.0))) - LOG_SCALE_MIN) / np.float32(127.0)
)


def scale_index(scales: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(scales, dtype=np.float32), 0.11, 16.0)
    v = (np.log(s) - LOG_SCALE_MIN) * LOG_STEP_RECIP
    return np.floor(v).astype(np.int32).clip(0, 127)


def fxp_conv1x1_np(x, w_int, w_scale, bias, x_scale):
    # x: [N,Cin,H,W], w_int: [Cout,Cin,1,1]
    n, cin, h, w = x.shape
    cout = w_int.shape[0]
    inv = np.float32(1.0) / np.float32(x_scale)
    xq = np.rint(x * inv).clip(-32768, 32767).astype(np.int16)
    # einsum over cin: int32 accumulate then dequant
    acc = np.zeros((n, cout, h, w), dtype=np.int32)
    w2 = w_int.reshape(cout, cin)
    for oc in range(cout):
        # sum_ic xq[:,ic] * w[oc,ic]
        acc[:, oc] = np.tensordot(xq.astype(np.int32), w2[oc].astype(np.int32), axes=([1], [0]))
    scale = (np.float32(x_scale) * w_scale.astype(np.float32)).reshape(1, cout, 1, 1)
    b = bias.astype(np.float32).reshape(1, cout, 1, 1)
    return acc.astype(np.float32) * scale + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=os.path.join(ONNX_DIR, "models", "y_prior_fusion.onnx"))
    ap.add_argument("--fxp-dir", default=os.path.join(ONNX_DIR, "models_fxp"))
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    args = ap.parse_args()

    meta = np.load(os.path.join(args.fxp_dir, "fxp_prior_fusion_meta.npz"))
    x_scale = float(meta["x_scale"])

    m = onnx.load(args.fp32)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    last = [n for n in m.graph.node if n.op_type == "Conv"][-1]
    w = inits[last.input[1]].astype(np.float32)
    b = inits[last.input[2]].astype(np.float32)
    mid_name = last.input[0]

    # Expose intermediate for FP32 path.
    g = onnx.load(args.fp32)
    if mid_name not in {o.name for o in g.graph.output}:
        from onnx import TensorProto, helper
        g.graph.output.append(
            helper.make_tensor_value_info(mid_name, TensorProto.FLOAT, ["N", "C", "H", "W"])
        )
    sess = ort.InferenceSession(g.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 256, args.h, args.w), dtype=np.float32)
    outs = sess.run(None, {in_name: x})
    by = dict(zip(out_names, outs))
    act = by[mid_name]
    y_fp32 = by["out0"]

    # Quantize weights same as export
    cout = w.shape[0]
    w_int = np.zeros_like(w, dtype=np.int8)
    w_scale = np.zeros((cout,), np.float32)
    for c in range(cout):
        a = max(float(np.max(np.abs(w[c]))), 1e-12)
        s = a / 127.0
        w_scale[c] = s
        w_int[c] = np.rint(w[c] / s).clip(-127, 127).astype(np.int8)

    y_fxp = fxp_conv1x1_np(act, w_int, w_scale, b, x_scale)
    err = np.abs(y_fxp - y_fp32)
    print(f"x_scale={x_scale:.8g}")
    print(f"output max|err|={err.max():.6g}  mean|err|={err.mean():.6g}")

    # scales are channels [2, 2+256)
    s_fp = y_fp32[0, 2:2 + 256]
    s_fx = y_fxp[0, 2:2 + 256]
    i_fp = scale_index(s_fp)
    i_fx = scale_index(s_fx)
    agree = float(np.mean(i_fp == i_fx)) * 100.0
    print(f"scale-index agree={agree:.3f}%  mismatches={int(np.sum(i_fp != i_fx))}/{i_fp.size}")
    print(f"scale max|err|={np.max(np.abs(s_fx - s_fp)):.6g}")


if __name__ == "__main__":
    main()
