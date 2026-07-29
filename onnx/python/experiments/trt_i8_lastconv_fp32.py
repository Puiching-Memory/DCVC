#!/usr/bin/env python3
"""Make the FINAL conv(s) of a QDQ INT8 model full-precision (graph surgery).

For every graph output: walk back through elementwise nodes to the producing
conv(s), then for each such conv
  - remove the Q/DQ pair on its activation input (input stays FP32),
  - fold the weight Q/DQ to FP32 (weights stay on their int8 grid values but
    are stored/computed as FP32),
  - remove Q/DQ pairs on the output chain to the graph output.

The result: last conv computes in FP32 from FP32 activations; everything
upstream keeps its INT8 QDQ (TRT fusion intact, unlike nodes_to_exclude).

Usage:
  trt_i8_lastconv_fp32.py --in m.onnx --out m2.onnx [--depth 1]
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--depth", type=int, default=1,
                    help="how many conv layers back from the output to free")
    args = ap.parse_args()

    m = onnx.load(args.src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    init_arr = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    folded_cache = {}

    def fold_weight_dq(dq_name):
        """If dq_name is produced by DQ<-Q<-fp32 init, fold to fp32 init; return new tensor name."""
        dqn = prod.get(dq_name)
        if dqn is None or dqn.op_type != "DequantizeLinear":
            return None
        qn = prod.get(dqn.input[0])
        if qn is None or qn.op_type != "QuantizeLinear":
            return None
        w = init_arr.get(qn.input[0])
        if w is None or w.dtype != np.float32:
            return None
        if qn.input[0] in folded_cache:
            return folded_cache[qn.input[0]]
        s = init_arr[qn.input[1]]
        zp = init_arr.get(qn.input[2])
        ax = next((a.i for a in qn.attribute if a.name == "axis"), 1)
        shp = [1] * w.ndim
        if s.size > 1:
            shp[ax] = -1
        s_b = s.reshape(shp)
        zp_b = zp.reshape(shp) if zp is not None else 0
        wd = (np.clip(np.round(w / s_b) + (zp_b if isinstance(zp_b, np.ndarray) else 0),
                      -128, 127) - zp_b) * s_b
        ni = numpy_helper.from_array(wd.astype(np.float32), qn.input[0] + "_folded")
        g.initializer.extend([ni])
        init_arr[ni.name] = wd.astype(np.float32)
        folded_cache[qn.input[0]] = ni.name
        return ni.name

    # find output convs: walk back from each graph output through non-conv nodes
    target_convs = set()
    for out in g.output:
        stack = [out.name]
        seen = set()
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            n = prod.get(t)
            if n is None:
                continue
            if n.op_type == "Conv":
                target_convs.add(n.name)
            else:
                stack.extend(n.input)

    # expand to --depth conv layers (linear back-walk through Q/DQ/elementwise)
    for _ in range(args.depth - 1):
        extra = set()
        for name in list(target_convs):
            n = next(x for x in g.node if x.name == name)
            t = n.input[0]
            seen = set()
            while True:
                if t in seen:
                    break
                seen.add(t)
                p = prod.get(t)
                if p is None:
                    break
                if p.op_type == "Conv":
                    extra.add(p.name)
                    break
                if len(p.input) == 0:
                    break
                t = p.input[0]
        target_convs |= extra

    drop = set()
    rewire = {}
    for name in target_convs:
        n = next(x for x in g.node if x.name == name)
        # 1) activation input: remove Q/DQ pair (keep producer output fp32)
        t = n.input[0]
        dqn = prod.get(t)
        if dqn is not None and dqn.op_type == "DequantizeLinear":
            qn = prod.get(dqn.input[0])
            if qn is not None and qn.op_type == "QuantizeLinear":
                rewire[dqn.output[0]] = qn.input[0]
                drop.add(qn.name)
                drop.add(dqn.name)
        # 2) weight: fold DQ to fp32 init and drop the weight Q/DQ nodes
        dqn_w = prod.get(n.input[1])
        new_w = fold_weight_dq(n.input[1])
        if new_w:
            rewire[n.input[1]] = new_w
            if dqn_w is not None:
                drop.add(dqn_w.name)
                qn_w = prod.get(dqn_w.input[0])
                if qn_w is not None and qn_w.op_type == "QuantizeLinear":
                    drop.add(qn_w.name)
    # 3) output chain: for graph outputs, remove trailing Q/DQ
    out_rewire = {}
    for out in g.output:
        n = prod.get(out.name)
        chain = []
        while n is not None and n.op_type in ("DequantizeLinear", "QuantizeLinear"):
            chain.append(n)
            n = prod.get(n.input[0])
        if chain and n is not None:
            for c in chain:
                drop.add(c.name)
            out_rewire[chain[-1].input[0]] = out.name

    nodes = []
    for n in g.node:
        if n.name in drop:
            continue
        ins = [rewire.get(t, t) for t in n.input]
        outs = [out_rewire.get(o, o) for o in n.output]
        if ins != list(n.input) or outs != list(n.output):
            n = onnx.helper.make_node(n.op_type, ins, outs, name=n.name,
                                      **{a.name: onnx.helper.get_attribute_value(a)
                                         for a in n.attribute})
        nodes.append(n)
    g.ClearField("node")
    g.node.extend(nodes)

    # drop unreferenced int8/quant initializers (avoid TRT import issues)
    referenced = set()
    for n in g.node:
        referenced.update(n.input)
    gnames = {i.name for i in g.input}
    keep = [i for i in g.initializer if i.name in referenced or i.name in gnames]
    g.ClearField("initializer")
    g.initializer.extend(keep)

    onnx.checker.check_model(m)
    onnx.save(m, args.dst)
    print(f"freed {len(target_convs)} convs: {sorted(target_convs)} -> {args.dst}")


if __name__ == "__main__":
    main()
