#!/usr/bin/env python3
"""Cross-Layer Equalization (CLE) for ONNX models — data-free weight transform.

CLE equalizes per-channel weight ranges across consecutive Conv layers so that
per-tensor activation quantization sees a more uniform range distribution.  It
is exact in FP32: the mathematical output of the network is unchanged.

Only Conv->Conv chains with no activation (or a positively-homogeneous
activation like ReLU) between them are equalizable.  In the DCVC-RT
DepthConv decomposition the pattern

    Conv(dc.2, depthwise) -> Conv(dc.3, pointwise)

and

    Conv(adaptor) -> Conv(dc.0)

are direct Conv->Conv connections (no op between them), so CLE applies cleanly.
The ws_relu (x * sigmoid(4x)) activations between blocks are NOT homogeneous,
so block-boundary pairs are skipped.

Algorithm (per pair conv1 -> conv2, sharing C channels):
    R1[c] = max(abs(W1[c]))        per output-channel of conv1
    R2[c] = max(abs(W2[:,c]))      per input-channel of conv2
    S[c]  = sqrt(R1[c] / R2[c])    equalizing scale
    W1[c] /= S[c];  W2[:,c] *= S[c];  B1[c] /= S[c] (if bias)

The equalized range for every channel becomes sqrt(R1[c] * R2[c]).

Reference: Nagel et al., "A White Paper on Neural Network Quantization"
(Qualcomm AI Research / AIMET).
"""
import argparse
import os

import numpy as np
import onnx
import onnx.numpy_helper as nh
import onnxruntime as ort

REPO = os.path.dirname(os.path.abspath(__file__))

TARGET_NETS = [
    "hyper_dec",
    "y_prior_fusion",
    "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1",
    "y_spatial_prior_adaptor_2",
    "y_spatial_prior_adaptor_3",
    "y_spatial_prior",
]


def _find_conv_conv_pairs(graph):
    """Return [(conv1_node, conv2_node)] for directly-connected Conv->Conv.

    Only pairs where conv1's output is consumed SOLELY by conv2 qualify. If the
    output also feeds a residual/Add/skip connection, scaling it would corrupt
    that branch — the classic CLE constraint.
    """
    producer = {}
    consumer_count = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node
    for node in graph.node:
        for inp in node.input:
            consumer_count[inp] = consumer_count.get(inp, 0) + 1
    pairs = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        src = producer.get(node.input[0])
        if src is not None and src.op_type == "Conv":
            if consumer_count.get(src.output[0], 0) == 1:
                pairs.append((src, node))
    return pairs


def _equalize_pair(graph, conv1, conv2, eps=1e-8):
    """Equalize one Conv->Conv pair in-place on *graph*.  Returns True if applied."""
    inits = {i.name: i for i in graph.initializer}
    if conv1.input[1] not in inits or conv2.input[1] not in inits:
        return False
    w1 = nh.to_array(inits[conv1.input[1]]).copy()
    w2 = nh.to_array(inits[conv2.input[1]]).copy()
    c_shared = w1.shape[0]
    if w2.shape[1] != c_shared:
        return False

    r1 = np.max(np.abs(w1), axis=tuple(range(1, w1.ndim))).astype(np.float64)
    r2 = np.max(np.abs(w2), axis=(0,) + tuple(range(2, w2.ndim))).astype(np.float64)
    s = np.sqrt(r1 / np.maximum(r2, eps))
    s = np.clip(s, 1e-6, 1e6)
    s = s.astype(w1.dtype)

    w1 /= s.reshape(c_shared, *([1] * (w1.ndim - 1)))
    w2 *= s.reshape(1, c_shared, *([1] * (w2.ndim - 2)))

    _replace_initializer(graph, conv1.input[1], w1)
    _replace_initializer(graph, conv2.input[1], w2)

    if len(conv1.input) > 2 and conv1.input[2] in inits:
        b1 = nh.to_array(inits[conv1.input[2]]).copy()
        b1 /= s
        _replace_initializer(graph, conv1.input[2], b1)
    return True


def _replace_initializer(graph, name, array):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            graph.initializer[i].CopyFrom(nh.from_array(np.ascontiguousarray(array), name=name))
            return
    graph.initializer.append(nh.from_array(np.ascontiguousarray(array), name=name))


def cle_equalize_model(model_path, out_path, iterations=2):
    """Apply CLE to an ONNX model and save.  Returns (n_pairs, n_iters)."""
    model = onnx.load(model_path)
    graph = model.graph
    total = 0
    for _ in range(iterations):
        pairs = _find_conv_conv_pairs(graph)
        if not pairs:
            break
        applied = 0
        for conv1, conv2 in pairs:
            if _equalize_pair(graph, conv1, conv2):
                applied += 1
        total += applied
    onnx.save(model, out_path)
    return total


def main():
    ap = argparse.ArgumentParser(description="Apply CLE to DCVC-RT entropy-parameter nets")
    ap.add_argument("--models-dir", default=os.path.normpath(os.path.join(REPO, "..", "models")))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_cle")))
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--verify", action="store_true", default=True,
                    help="verify FP32 output unchanged after CLE")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in TARGET_NETS:
        src = os.path.join(args.models_dir, name + ".onnx")
        dst = os.path.join(args.out_dir, name + ".onnx")
        if not os.path.exists(src):
            print(f"{name}: SKIP (not found)")
            continue
        n = cle_equalize_model(src, dst, args.iterations)
        if args.verify and n > 0:
            import numpy as np
            in_dims = [d.dim_value for d in onnx.load(src).graph.input[0].type.tensor_type.shape.dim]
            ic = in_dims[1] if len(in_dims) > 1 and in_dims[1] > 0 else 256
            inp = np.random.randn(1, ic, 16, 16).astype(np.float32)
            sess_o = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
            sess_c = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
            iname = sess_o.get_inputs()[0].name
            o_ref = sess_o.run(None, {iname: inp})[0]
            o_cle = sess_c.run(None, {iname: inp})[0]
            md = float(np.abs(o_ref - o_cle).max())
            status = "OK" if md < 1e-3 else "FAILED"
            print(f"{name:35s} pairs_equalized={n:3d}  maxdiff={md:.2e}  {status}")
        else:
            print(f"{name:35s} pairs_equalized={n:3d}")
    # copy non-target files
    for f in os.listdir(args.models_dir):
        s = os.path.join(args.models_dir, f)
        d = os.path.join(args.out_dir, f)
        if os.path.isfile(s) and not os.path.exists(d) and not f.endswith(".onnx"):
            import shutil
            shutil.copy2(s, d)


if __name__ == "__main__":
    raise SystemExit(main())
