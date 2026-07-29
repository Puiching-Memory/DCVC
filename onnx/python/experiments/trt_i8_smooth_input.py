#!/usr/bin/env python3
"""Per-channel input smoothing for TRT INT8 PTQ (activation outlier fix).

Problem: per-tensor INT8 activation quantization on heavy-tailed, per-channel-
scaled inputs (e.g. y_hat = y * qdec) gives the largest channel full range and
leaves small channels a few quantization levels -> dominant RD loss
(measured: intra_synthesis alone = -4.65 dB @qp32).

Fix (SmoothQuant-style, exact weight fold): normalize the input per channel
    x' = x / s_c,  s_c = max|x[:,c]| over the calibration set
and fold s_c into the FIRST conv's weights: W'[:,c,:,:] = W[:,c,:,:] * s_c.
The network is mathematically identical in FP32; after normalization every
channel spans ~[-1,1], so a per-tensor INT8 activation scale resolves all
channels equally.

Only valid when the input tensor has exactly one consumer (a Conv); asserted.

Usage:
  .venv/bin/python trt_i8_smooth_input.py --net intra_synthesis \
      --calib calib_trt_i8/intra_synthesis --in-tensor in0 \
      --src ../models_fp32 --dst ../models_smooth
"""
import argparse
import glob
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--calib", required=True, help="dir with sample_*.npz")
    ap.add_argument("--in-tensor", default="in0")
    ap.add_argument("--src", default=os.path.normpath(os.path.join(REPO, "..", "models_fp32")))
    ap.add_argument("--dst", default=os.path.normpath(os.path.join(REPO, "..", "models_smooth")))
    args = ap.parse_args()

    # 1) per-channel max over calib set
    smax = None
    for f in sorted(glob.glob(os.path.join(args.calib, "sample_*.npz"))):
        z = np.load(f)
        x = np.abs(z[args.in_tensor].astype(np.float64))
        m = x.max(axis=(0, 2, 3))
        smax = m if smax is None else np.maximum(smax, m)
    assert smax is not None and (smax > 0).all(), "empty calib or dead channel"
    s = np.maximum(smax, 1e-3).astype(np.float32)
    C = s.shape[0]
    print(f"channels={C} scale range=[{s.min():.4g}, {s.max():.4g}] ratio={s.max()/s.min():.1f}")

    m = onnx.load(os.path.join(args.src, args.net + ".onnx"))
    g = m.graph
    init = {i.name: i for i in g.initializer}

    consumers = [(i, n) for i, n in enumerate(g.node) if args.in_tensor in n.input]
    assert len(consumers) == 1, f"{args.in_tensor} has {len(consumers)} consumers"
    conv_idx, conv = consumers[0]
    assert conv.op_type == "Conv", f"consumer is {conv.op_type}"

    # 2) fold s into conv weights (input-channel axis = 1)
    w = numpy_helper.to_array(init[conv.input[1]])
    assert w.shape[1] == C, f"weight in_ch={w.shape[1]} != {C}"
    w_new = (w * s.reshape(1, C, 1, 1)).astype(np.float32)
    folded = numpy_helper.from_array(w_new, conv.input[1] + "_smooth")
    g.initializer.extend([folded])
    conv.input[1] = folded.name

    # 3) insert Mul(in0, 1/s) before the conv
    inv = numpy_helper.from_array((1.0 / s).reshape(1, C, 1, 1).astype(np.float32),
                                  args.in_tensor + "_invscale")
    g.initializer.extend([inv])
    normed = args.in_tensor + "_normed"
    mul = helper.make_node("Mul", [args.in_tensor, inv.name], [normed],
                           name=args.net + "_input_smooth")
    conv.input[0] = normed
    g.node.insert(conv_idx, mul)

    os.makedirs(args.dst, exist_ok=True)
    out = os.path.join(args.dst, args.net + ".onnx")
    onnx.checker.check_model(m)
    onnx.save(m, out)
    print("->", out)


if __name__ == "__main__":
    main()
