#!/usr/bin/env python3
"""Replace y_prior_fusion's final 1x1 Conv with com.dcvc::FxpConv1x1.

PoC for bit-exact entropy-parameter inference via an ORT custom op:
  float X -> int16 act / int8 per-channel weights -> float Y

Usage:
    uv run python python/fxp_export_prior_fusion.py
    uv run python python/fxp_export_prior_fusion.py --out-dir models_fxp
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))
DEFAULT_SRC = os.path.join(ONNX_DIR, "models", "y_prior_fusion.onnx")


def _find_last_conv(graph):
    convs = [n for n in graph.node if n.op_type == "Conv"]
    if not convs:
        raise RuntimeError("no Conv nodes")
    return convs[-1]


def _init_map(graph):
    return {i.name: i for i in graph.initializer}


def calibrate_x_scale(model_path: str, last_conv_input: str, samples: int, h: int, w: int) -> float:
    """Max-abs activation scale for int16: x_scale = max|x| / 32767."""
    m = onnx.load(model_path)
    # Expose the pre-final-conv tensor as an extra graph output for calib.
    existing = {o.name for o in m.graph.output}
    if last_conv_input not in existing:
        # Find value_info or invent a float tensor info.
        vi = None
        for cand in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
            if cand.name == last_conv_input:
                vi = cand
                break
        if vi is None:
            vi = helper.make_tensor_value_info(last_conv_input, TensorProto.FLOAT, ["N", "C", "H", "W"])
        m.graph.output.append(vi)

    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    # Prefer the intermediate; fall back to scanning.
    target = last_conv_input if last_conv_input in out_names else out_names[-1]

    absmax = 0.0
    rng = np.random.default_rng(0)
    for _ in range(samples):
        x = rng.standard_normal((1, 256, h, w), dtype=np.float32)
        outs = sess.run(None, {in_name: x})
        # Map by name
        by_name = {n: v for n, v in zip(out_names, outs)}
        act = by_name[target]
        absmax = max(absmax, float(np.max(np.abs(act))))

    if absmax < 1e-8:
        absmax = 1.0
    return absmax / 32767.0


def quantize_weights(w: np.ndarray):
    """Per-output-channel int8 symmetric quantization. w: [Cout, Cin, 1, 1]."""
    cout = w.shape[0]
    w_int = np.zeros_like(w, dtype=np.int8)
    w_scale = np.zeros((cout,), dtype=np.float32)
    for c in range(cout):
        a = float(np.max(np.abs(w[c])))
        if a < 1e-12:
            a = 1.0
        s = a / 127.0
        w_scale[c] = s
        q = np.rint(w[c] / s).clip(-127, 127).astype(np.int8)
        w_int[c] = q
    return w_int, w_scale


def replace_last_conv(src_path: str, dst_path: str, x_scale: float):
    m = onnx.load(src_path)
    g = m.graph
    last = _find_last_conv(g)
    inits = _init_map(g)

    x_name = last.input[0]
    w_name = last.input[1]
    b_name = last.input[2] if len(last.input) > 2 else None
    y_name = last.output[0]

    w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
    b = numpy_helper.to_array(inits[b_name]).astype(np.float32) if b_name else np.zeros(w.shape[0], np.float32)
    w_int, w_scale = quantize_weights(w)

    w_int_name = "fxp_W_int"
    w_scale_name = "fxp_W_scale"
    b_name_new = "fxp_B"

    # Drop FP32 weight/bias initializers for the replaced conv (keep others).
    keep = []
    drop = {w_name, b_name} if b_name else {w_name}
    for init in g.initializer:
        if init.name not in drop:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.append(numpy_helper.from_array(w_int, w_int_name))
    g.initializer.append(numpy_helper.from_array(w_scale, w_scale_name))
    g.initializer.append(numpy_helper.from_array(b.astype(np.float32), b_name_new))

    # Remove the last Conv node.
    nodes = [n for n in g.node if n is not last]
    del g.node[:]
    g.node.extend(nodes)

    fxp = helper.make_node(
        "FxpConv1x1",
        inputs=[x_name, w_int_name, b_name_new, w_scale_name],
        outputs=[y_name],
        name="fxp_conv1x1_last",
        domain="com.dcvc",
        x_scale=float(x_scale),
    )
    g.node.append(fxp)

    # Ensure custom domain opset is declared.
    domains = {o.domain: o for o in m.opset_import}
    if "com.dcvc" not in domains:
        m.opset_import.append(helper.make_opsetid("com.dcvc", 1))

    onnx.save(m, dst_path)
    return w.shape, float(x_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out-dir", default=os.path.join(ONNX_DIR, "models_fxp"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--x-scale", type=float, default=None,
                    help="skip calibration and use this activation scale")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy full model dir so the fxp prior can sit alongside unchanged nets.
    src_dir = os.path.dirname(args.src)
    if os.path.isdir(src_dir):
        for name in os.listdir(src_dir):
            s = os.path.join(src_dir, name)
            d = os.path.join(args.out_dir, name)
            if os.path.isfile(s) and not os.path.exists(d):
                shutil.copy2(s, d)

    m = onnx.load(args.src)
    last = _find_last_conv(m.graph)
    last_in = last.input[0]

    if args.x_scale is not None:
        x_scale = float(args.x_scale)
        print(f"using provided x_scale={x_scale}")
    else:
        print(f"calibrating on {args.samples} random {args.h}x{args.w} tensors…")
        x_scale = calibrate_x_scale(args.src, last_in, args.samples, args.h, args.w)
        print(f"calibrated x_scale={x_scale:.8g}")

    dst = os.path.join(args.out_dir, "y_prior_fusion.onnx")
    shape, xs = replace_last_conv(args.src, dst, x_scale)
    meta = os.path.join(args.out_dir, "fxp_prior_fusion_meta.npz")
    np.savez(meta, x_scale=np.float32(xs), w_shape=np.array(shape))
    print(f"wrote {dst}")
    print(f"wrote {meta}")
    print(f"W shape {shape}, x_scale={xs:.8g}")


if __name__ == "__main__":
    main()
