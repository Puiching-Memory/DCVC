#!/usr/bin/env python3
"""Convert INT16 QDQ ONNX models to FP32 emulation for TensorRT.

TRT has no INT16 dtype (parser rejects int16 initializers), so each
QuantizeLinear(int16) is rewritten as an FP32 subgraph that reproduces the
exact int16 quantization numerics:

    q   = clip(round(x / scale), -32768, 32767)      (zp=0 fast path)
    emu = q * scale

Div/Round/Clip/Mul are all IEEE-exact elementwise ops, bit-identical between
ORT CPU and TRT, so the quantized grid values match ORT's int16 execution
exactly; only conv accumulation order differs (absorbed by the 1/512 grid).

DequantizeLinear handling:
  - on int16 weight initializers: constant-fold to FP32 weights;
  - on the output of an emulated Q: bypass (consumers rewired to Q-emu output).

Usage:
  .venv/bin/python onnx/python/trt_i16_emulate.py [--src ../models_int16]
      [--dst ../models_int16_trtemu] [--nets hyper_dec ...]
"""
import argparse
import glob
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = os.path.dirname(os.path.abspath(__file__))
QMIN, QMAX = -32768.0, 32767.0


def convert(src_path, dst_path):
    m = onnx.load(src_path)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    init_arr = {i.name: numpy_helper.to_array(i) for i in g.initializer}

    # consumer map: tensor -> list of node indices
    consumers = {}
    for idx, n in enumerate(g.node):
        for t in n.input:
            consumers.setdefault(t, []).append(idx)

    new_nodes = []
    skip = set()          # node indices to drop
    rewire = {}           # tensor name -> replacement tensor name
    added_inits = []

    def scalar_init(name, val):
        t = numpy_helper.from_array(np.array(val, dtype=np.float32), name)
        added_inits.append(t)
        return name

    for idx, n in enumerate(g.node):
        if n.op_type == "DequantizeLinear":
            x_name = n.input[0]
            if x_name in init and init_arr[x_name].dtype in (np.int16, np.int32):
                # weight/bias DQ -> fold to fp32
                s = init_arr[n.input[1]]
                zp = init_arr.get(n.input[2])
                zp = zp if zp is not None else np.zeros_like(s, dtype=np.int16)
                axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
                w = init_arr[x_name].astype(np.float32)
                shp = [1] * w.ndim
                if s.size > 1:
                    shp[axis] = -1
                wdq = (w - zp.astype(np.float32).reshape(shp)) * s.reshape(shp)
                folded = numpy_helper.from_array(
                    wdq.astype(np.float32), x_name + "_folded_f32")
                added_inits.append(folded)
                rewire[n.output[0]] = folded.name
                skip.add(idx)

    emu_chains = {}       # node idx -> list of emu nodes (inserted in place)

    for idx, n in enumerate(g.node):
        if n.op_type != "QuantizeLinear":
            continue
        x, s_name = n.input[0], n.input[1]
        zp_name = n.input[2] if len(n.input) > 2 else None
        zp = init_arr.get(zp_name) if zp_name else None
        zp_zero = zp is None or not np.any(zp)
        base = n.output[0]
        div = base + "_emu_div"
        rnd = base + "_emu_rnd"
        clp = base + "_emu_clp"
        sub = base + "_emu_sub"
        out = base + "_emu"
        chain = [helper.make_node("Div", [x, s_name], [div], name=base + "_emu_Div"),
                 helper.make_node("Round", [div], [rnd], name=base + "_emu_Round")]
        mn = scalar_init(base + "_qmin", QMIN if zp_zero else QMIN - float(zp.min()))
        mx = scalar_init(base + "_qmax", QMAX if zp_zero else QMAX - float(zp.max()))
        chain.append(helper.make_node("Clip", [rnd, mn, mx], [clp], name=base + "_emu_Clip"))
        if zp_zero:
            chain.append(helper.make_node("Mul", [clp, s_name], [out], name=base + "_emu_Mul"))
        else:
            zpf = numpy_helper.from_array(zp.astype(np.float32), zp_name + "_f32")
            added_inits.append(zpf)
            chain.append(helper.make_node("Add", [clp, zpf.name], [sub], name=base + "_emu_ZpAdd"))
            chain.append(helper.make_node("Sub", [sub, zpf.name], [sub + "b"], name=base + "_emu_ZpSub"))
            chain.append(helper.make_node("Mul", [sub + "b", s_name], [out], name=base + "_emu_Mul"))
        emu_chains[idx] = chain
        rewire[base] = out          # Q int16 output -> emu fp32
        skip.add(idx)

    # rebuild node list with rewiring, dropping replaced Q/DQ nodes and
    # inserting Q-emu chains at the original Q position (keeps topo order)
    final_nodes = []
    for idx, n in enumerate(g.node):
        if idx in emu_chains:
            final_nodes.extend(emu_chains[idx])
            continue
        if idx in skip:
            continue
        if n.op_type == "DequantizeLinear" and n.input[0] in rewire:
            skip.add(idx)
            rewire[n.output[0]] = rewire[n.input[0]]
            continue
        new_input = [rewire.get(t, t) for t in n.input]
        if new_input != list(n.input):
            n = helper.make_node(n.op_type, new_input, list(n.output), name=n.name,
                                 **{a.name: helper.get_attribute_value(a) for a in n.attribute})
        final_nodes.append(n)

    # fix graph outputs / value infos referencing rewired tensors
    for outs in (g.output,):
        for o in outs:
            if o.name in rewire:
                o.name = rewire[o.name]

    # drop initializers no longer referenced (int16/int32 weights, zps, scales
    # of removed Q nodes) so TRT doesn't hit unsupported-dtype imports
    referenced = set()
    for n in final_nodes:
        referenced.update(n.input)
    ginput_names = {i.name for i in g.input}
    keep_inits = [i for i in g.initializer
                  if i.name in referenced or i.name in ginput_names]

    g.ClearField("node")
    g.node.extend(final_nodes)
    g.ClearField("initializer")
    g.initializer.extend(keep_inits + added_inits)

    # remove now-dangling value infos for int16 tensors (optional cleanup)
    onnx.checker.check_model(m)
    onnx.save(m, dst_path)
    n_q = sum(1 for n in new_nodes if n.op_type == "Round")
    print(f"{os.path.basename(dst_path)}: Q-emulated={n_q} nodes={len(g.node)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(os.path.join(REPO, "..", "models_int16")))
    ap.add_argument("--dst", default=os.path.normpath(os.path.join(REPO, "..", "models_int16_trtemu")))
    ap.add_argument("--nets", nargs="+", default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in glob.glob(os.path.join(args.src, "*.npy")):
        dst = os.path.join(args.dst, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)

    nets = args.nets or [os.path.splitext(os.path.basename(p))[0]
                         for p in glob.glob(os.path.join(args.src, "*.onnx"))]
    for net in nets:
        convert(os.path.join(args.src, net + ".onnx"),
                os.path.join(args.dst, net + ".onnx"))
    print("done ->", args.dst)


if __name__ == "__main__":
    main()
