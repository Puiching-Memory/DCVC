#!/usr/bin/env python3
"""Fuse adjacent conv -> FxpWsRelu node pairs into a single fused custom op
(com.dcvc domain).

Two conv variants are supported (same domain ``com.dcvc``):
  * FxpConv   (int16) -> FxpConvWsRelu   inputs [X, W, B, Ws, Lut]
  * FxpConvI8 (int8)  -> FxpConvI8WsRelu inputs [X, W, B, Ws, Wcomp, Lut]

This is a PURE GRAPH REWRITE: no re-quantization or calibration is performed,
so the fused graph computes bit-identical values to the unfused one (the C++
fused kernel runs the exact same conv + wsrelu math, just without the
intermediate tensor round-trip through main memory).

A pair (conv, wsrelu) is fusible iff:
  * wsrelu.input[0] is produced by a FxpConv / FxpConvI8 node, AND
  * that conv output has exactly ONE consumer (the wsrelu) -- not shared with
    any other node or a graph output, AND
  * the conv matches one of the C++ kernel's fast paths:
      - 1x1, group=1, pads=[0,0,0,0], strides=[1,1], OR
      - depthwise 3x3, group==Cin==Cout, pads=[1,1,1,1], strides=[1,1]

Usage:
    python fuse_conv_wsrelu.py --src-dir models --out-dir models_fused
    python fuse_conv_wsrelu.py --src-dir models --only y_prior_fusion hyper_dec
"""
import argparse
import copy
import glob
import os
import shutil
import sys

import onnx
from onnx import helper, checker

CONV_OP = "FxpConv"
WSRELU_OP = "FxpWsRelu"
FUSED_OP = "FxpConvWsRelu"
DOMAIN = "com.dcvc"
# conv attributes to copy verbatim onto the fused node
CONV_ATTRS = ("group", "pads", "strides", "x_scale", "act_bits")

# Conv variants that can be fused with a trailing FxpWsRelu.  Each entry maps
# conv op_type -> (fused_op_name, n_conv_inputs).  The fused node receives the
# first ``n_conv_inputs`` conv inputs verbatim, followed by the wsrelu LUT
# (ws.input[1]).  wsrelu_x_scale and CONV_ATTRS are identical for both.
CONV_FUSION = {
    "FxpConv":   ("FxpConvWsRelu", 4),    # [X, W, B, Ws]
    "FxpConvI8": ("FxpConvI8WsRelu", 5),  # [X, W, B, Ws, Wcomp]
}


def _attr_value_int(node, name, default):
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.INT:
            return int(a.i)
    return default


def _attr_value_ints(node, name, default):
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.INTS:
            return [int(x) for x in a.ints]
    return list(default)


def _attr_value_float(node, name, default=None):
    for a in node.attribute:
        if a.name == name and a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
    return default


def _initializer_dims(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _is_fusable(conv, graph):
    """Return True iff conv matches a C++ FxpConvWsRelu fast path."""
    grp = _attr_value_int(conv, "group", 1)
    pads = _attr_value_ints(conv, "pads", [0, 0, 0, 0])
    strides = _attr_value_ints(conv, "strides", [1, 1])
    w = _initializer_dims(graph, conv.input[1])  # [Cout, Cin/group, kh, kw]
    if not w or len(w) != 4:
        return False
    cout, cin_per_g, kh, kw = w
    cin = cin_per_g * grp
    is_1x1 = (kh == 1 and kw == 1 and grp == 1
              and pads == [0, 0, 0, 0] and strides == [1, 1])
    is_dw = (kh == 3 and kw == 3 and grp == cin and cin == cout
             and pads == [1, 1, 1, 1] and strides == [1, 1])
    return is_1x1 or is_dw


def _consumer_counts(graph):
    """tensor_name -> number of consumers (node inputs + graph outputs)."""
    cnt = {}
    for n in graph.node:
        for i in n.input:
            if i:
                cnt[i] = cnt.get(i, 0) + 1
    for o in graph.output:
        if o.name:
            cnt[o.name] = cnt.get(o.name, 0) + 1
    return cnt


def fuse_model(model):
    """Fuse one model in place. Returns number of pairs fused."""
    g = model.graph

    producers = {}
    for n in g.node:
        for o in n.output:
            if o:
                producers[o] = n
    counts = _consumer_counts(g)

    # Map: wsrelu node id -> (conv node, wsrelu node); mark convs consumed by a
    # fusable wsrelu as "to remove".
    fuse_conv = {}      # id(conv) -> wsrelu
    fuse_wsrelu = {}    # id(wsrelu) -> conv
    fused = 0
    for ws in g.node:
        if ws.op_type != WSRELU_OP:
            continue
        src = ws.input[0]
        conv = producers.get(src)
        if conv is None or conv.op_type not in CONV_FUSION:
            continue
        if counts.get(src, 0) != 1:
            continue  # conv output shared -> skip
        if not _is_fusable(conv, g):
            continue
        fuse_conv[id(conv)] = ws
        fuse_wsrelu[id(ws)] = conv
        fused += 1

    if fused == 0:
        return 0

    used_names = {n.name for n in g.node if n.name}
    new_nodes = []
    for n in g.node:
        if id(n) in fuse_wsrelu:
            # this wsrelu is absorbed into its conv's fused node (emitted there)
            continue
        if id(n) in fuse_conv:
            ws = fuse_conv[id(n)]
            fused_op, n_conv_inputs = CONV_FUSION[n.op_type]
            # copy raw conv attributes verbatim (preserves INT/INTS/FLOAT types)
            attrs = []
            for a in n.attribute:
                if a.name in CONV_ATTRS:
                    attrs.append(copy.deepcopy(a))
            # wsrelu_x_scale: wsrelu x_scale, falling back to conv x_scale
            wsx = _attr_value_float(ws, "x_scale",
                                    _attr_value_float(n, "x_scale"))
            attrs.append(helper.make_attribute("wsrelu_x_scale", float(wsx)))

            # conv inputs forwarded verbatim, then the wsrelu LUT
            inputs = list(n.input[:n_conv_inputs]) + [ws.input[1]]
            base = n.name or ws.name or "fxp_conv_wsrelu"
            name = base + "_fused"
            i = 1
            while name in used_names:
                name = "%s_fused%d" % (base, i)
                i += 1
            used_names.add(name)

            fused_node = helper.make_node(
                fused_op, inputs=inputs, outputs=[ws.output[0]],
                name=name, domain=DOMAIN)
            fused_node.attribute.extend(attrs)
            new_nodes.append(fused_node)
            continue
        new_nodes.append(n)

    del g.node[:]
    g.node.extend(new_nodes)

    # Drop value_info / outputs that are no longer produced by any node.
    produced = {o for n in g.node for o in n.output if o}
    g_inputs = {i.name for i in g.input}
    keep_vi = [v for v in g.value_info
               if v.name in produced or v.name in g_inputs]
    del g.value_info[:]
    g.value_info.extend(keep_vi)

    # Remove orphaned initializers (referenced by no node input nor graph output).
    referenced = set()
    for n in g.node:
        referenced.update(i for i in n.input if i)
    referenced.update(o.name for o in g.output)
    keep_init = [init for init in g.initializer if init.name in referenced]
    if len(keep_init) != len(g.initializer):
        del g.initializer[:]
        g.initializer.extend(keep_init)

    return fused


def scan_dir(d):
    """Return {basename: Counter of op_type} for every .onnx in d."""
    from collections import Counter
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.onnx"))):
        m = onnx.load(f)
        out[os.path.basename(f)] = Counter(n.op_type for n in m.graph.node)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default="models", help="source model dir")
    ap.add_argument("--out-dir", default="models_fused", help="output dir")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only process these basenames (with or without .onnx)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.only:
        wanted = {os.path.splitext(n)[0] for n in args.only}
        files = []
        for f in sorted(glob.glob(os.path.join(args.src_dir, "*.onnx"))):
            if os.path.splitext(os.path.basename(f))[0] in wanted:
                files.append(f)
    else:
        files = sorted(glob.glob(os.path.join(args.src_dir, "*.onnx")))

    total_pairs = 0
    nets_touched = 0
    for f in files:
        m = onnx.load(f)
        before = sum(1 for n in m.graph.node if n.op_type == WSRELU_OP)
        n = fuse_model(m)
        if n == 0:
            # nothing fused: still copy unchanged so out-dir is a complete pack
            out = os.path.join(args.out_dir, os.path.basename(f))
            onnx.save(m, out)
            continue
        checker.check_model(m)
        out = os.path.join(args.out_dir, os.path.basename(f))
        onnx.save(m, out)
        total_pairs += n
        nets_touched += 1
        print("  %-40s fused %2d/%d pairs -> %s"
              % (os.path.basename(f), n, before, out))

    # Copy non-.onnx companions (CDF tables, q_*.npy, ...) verbatim.
    for f in glob.glob(os.path.join(args.src_dir, "*")):
        if os.path.isdir(f):
            continue
        if f.endswith(".onnx"):
            continue
        shutil.copy2(f, os.path.join(args.out_dir, os.path.basename(f)))

    print("fused %d pairs across %d nets -> %s"
          % (total_pairs, nets_touched, args.out_dir))

    # Summary scan of the output dir.
    scan = scan_dir(args.out_dir)
    agg = {}
    for c in scan.values():
        for op, k in c.items():
            agg[op] = agg.get(op, 0) + k
    report_ops = ({CONV_OP, WSRELU_OP, FUSED_OP}
                  | set(CONV_FUSION)
                  | {fused_op for fused_op, _ in CONV_FUSION.values()})
    keep = {k: v for k, v in sorted(agg.items()) if k in report_ops}
    print("models_fused op counts:", keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
