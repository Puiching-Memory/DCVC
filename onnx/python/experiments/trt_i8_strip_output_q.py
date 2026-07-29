#!/usr/bin/env python3
"""Remove the output QuantizeLinear/DequantizeLinear of the final conv(s)
producing a model's graph output in an INT8 QDQ model.

Rationale: the fused entropy-param output (means/scales) needs ~11 bits of
resolution that per-tensor int8 cannot provide (means range +-45 -> step 0.35,
widening the rANS residual -> rate explosion). Keeping ONLY the output conv's
result in FP32 restores the precision while all upstream convs stay INT8.
Unlike modelopt nodes_to_exclude (which breaks TRT QDQ fusion globally), this
surgery keeps every other Q/DQ pair intact.

For each graph output tensor: walk back DQ -> Q -> conv, delete Q and DQ,
rewire conv output to the graph output name.

Usage:
  .venv/bin/python trt_i8_strip_output_q.py --in models_trt_i8/y_prior_fusion.onnx \
      --out models_v3/y_prior_fusion.onnx
"""
import argparse

import onnx


def strip(path_in, path_out):
    m = onnx.load(path_in)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    drop = set()
    rewire = {}
    for out in g.output:
        n = prod.get(out.name)
        # allow output directly from Q, DQ, or conv
        chain = []
        while n is not None and n.op_type in ("DequantizeLinear", "QuantizeLinear"):
            chain.append(n)
            n = prod.get(n.input[0])
        if n is None or n.op_type != "Conv":
            print(f"  output {out.name}: producer chain not conv-rooted, skipped")
            continue
        for c in chain:
            drop.add(c.name)
        rewire[n.output[0]] = out.name
        print(f"  output {out.name}: stripped {len(chain)} Q/DQ nodes after {n.name}")

    nodes = []
    for n in g.node:
        if n.name in drop:
            continue
        outs = [rewire.get(o, o) for o in n.output]
        if outs != list(n.output):
            n = onnx.helper.make_node(n.op_type, list(n.input), outs, name=n.name,
                                      **{a.name: onnx.helper.get_attribute_value(a)
                                         for a in n.attribute})
        nodes.append(n)
    g.ClearField("node")
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path_out)
    print("->", path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()
    strip(args.src, args.dst)


if __name__ == "__main__":
    main()
