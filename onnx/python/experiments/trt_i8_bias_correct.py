#!/usr/bin/env python3
"""Layer-wise empirical bias correction for INT8 QDQ models.

Activation quantization shifts the mean of every conv output; the shift
accumulates through the stack (measured: ~60% of the INT8 error variance of
intra_synthesis is systematic bias). This pass walks the convs in topological
order and adds the per-output-channel mean difference (FP32 vs INT8, measured
on the calibration inputs) into the int8 conv's fp32 bias, so each layer's
output re-centers on the FP32 reference before the error can compound.

Usage:
  .venv/bin/python trt_i8_bias_correct.py --net intra_synthesis \
      --fp32 ../models_smooth/intra_synthesis.onnx \
      --int8 ../models_v2/intra_synthesis.onnx \
      --calib calib_trt_i8/intra_synthesis --out /tmp/synth_bc.onnx
"""
import argparse
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, utils


def truncate(m, upto_idx, out_name, path):
    """Build a sub-model of nodes[0..upto_idx] with single output out_name.

    Nodes are topologically sorted in these exports, so a prefix slice is a
    valid graph. Initializers/value infos are carried over wholesale."""
    g = m.graph
    sub = onnx.helper.make_graph(
        list(g.node[: upto_idx + 1]), g.name + "_sub", list(g.input),
        [onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)],
        list(g.initializer))
    sub.value_info.extend(g.value_info)
    mm = onnx.helper.make_model(sub, producer_name="bias_correct",
                                opset_imports=list(m.opset_import))
    mm.ir_version = m.ir_version
    onnx.save(mm, path)
    return path


def load_calib(calib_dir):
    """Yield feed dicts. npz (multi-input) or npy (single input 'in0')."""
    files = sorted(glob.glob(os.path.join(calib_dir, "sample_*.npz"))) or \
            sorted(glob.glob(os.path.join(calib_dir, "*.npy")))
    for f in files:
        if f.endswith(".npz"):
            z = np.load(f)
            yield {k: z[k].astype(np.float32) for k in z.files}
        else:
            yield {"in0": np.load(f).astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-calib", type=int, default=8)
    args = ap.parse_args()

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    prov = ["CPUExecutionProvider"]

    mf = onnx.load(args.fp32)
    mq = onnx.load(args.int8)
    input_names = [i.name for i in mq.graph.input]
    calib = list(load_calib(args.calib))[: args.max_calib]

    qi = {i.name: i for i in mq.graph.initializer}
    # fp32 conv output name -> node index (same node names in both models)
    f_conv_idx = {n.output[0]: i for i, n in enumerate(mf.graph.node)
                  if n.op_type == "Conv"}
    n_fixed = 0
    for idx, node in enumerate(mq.graph.node):
        if node.op_type != "Conv":
            continue
        out = node.output[0]
        if out not in f_conv_idx:
            print(f"skip {node.name}: no fp32 counterpart")
            continue
        truncate(mf, f_conv_idx[out], out, "/tmp/bc_f.onnx")
        truncate(mq, idx, out, "/tmp/bc_q.onnx")
        sf = ort.InferenceSession("/tmp/bc_f.onnx", sess_options=opts, providers=prov)
        sq = ort.InferenceSession("/tmp/bc_q.onnx", sess_options=opts, providers=prov)
        diff = None
        for feed in calib:
            of = sf.run([out], feed)[0].astype(np.float64)
            oq = sq.run([out], feed)[0].astype(np.float64)
            d = (of - oq).mean(axis=(0, 2, 3))
            diff = d if diff is None else diff + d
        diff = (diff / len(calib)).astype(np.float32)  # [C_out]
        if len(node.input) >= 3:
            b = numpy_helper.to_array(qi[node.input[2]])
            assert b.shape == diff.shape, f"{node.name}: bias {b.shape} vs diff {diff.shape}"
            qi[node.input[2]].CopyFrom(numpy_helper.from_array(b + diff, node.input[2]))
            n_fixed += 1

    onnx.save(mq, args.out)
    print(f"bias-corrected {n_fixed} convs -> {args.out}")


if __name__ == "__main__":
    main()
