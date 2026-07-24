#!/usr/bin/env python3
"""Convert the FP32 source nets to the com.dcvc::FxpConvI8 int8 path.

Replaces:
  Conv (group==1)               -> com.dcvc::FxpConvI8  (int8 act * int8 w, VNNI)
  Conv (group!=1, depthwise)    -> com.dcvc::FxpConv    (int16 fallback; i8 unsupported)
  Mul(x,4)->Sigmoid->Mul(s,x)   -> com.dcvc::FxpWsRelu  (65536-entry int16 LUT, unchanged)

FxpConvI8 formula (runtime):
    x_u8   = quantize(x, x_scale) + 128        (uint8 activation offset for VNNI)
    y[oc]  = (sum_u8i8(x_u8, W) - Wcomp[oc]) * x_scale * Ws[oc] + B[oc]
    Wcomp[oc] = 128 * sum(W_int8[oc])           (cancels the +128 offset)

So y[oc] == sum(quantize(x,x_scale) * W_int8) * x_scale * Ws[oc] + B[oc],
i.e. per-channel int8 weight + per-tensor int8 activation (x_scale=absmax/127),
keeping the per-output-channel Ws that the per-tensor attempt could not.

Usage:
    python fxp_export_int8.py --src-dir models_fp32 --out-dir models_int8 \
        --calib-dir python/calib_int8 --percentile 100
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Reuse all the shared infrastructure from the int16 export rather than
# duplicating it (calib loading, WsRelu pattern finding, LUT baking, splits).
from fxp_export_entropy_nets import (
    REPO,
    ONNX_DIR,
    ENTROPY_NETS,
    NET_HW,
    NET_CIN,
    OUTPUT_SPLITS,
    _SINGLE,
    _attr_map,
    _find_wsrelu_patterns,
    _bake_wsrelu_lut,
    _collect_act_ranges,
    _load_calib_inputs,
    _quantize_weights,          # int16 per-channel (used for depthwise fallback)
    _make_fxp_conv_node,        # int16 FxpConv node (depthwise fallback)
)

INT8_LEVELS = 127.0          # activation quantization levels
INT16_LEVELS = 32767.0       # int16 act (depthwise fallback) + WsRelu LUT


def _quantize_weights_i8(w: np.ndarray):
    """Per-output-channel symmetric int8 quantization (scale=max|w|/127)."""
    cout = w.shape[0]
    w_int = np.zeros_like(w, dtype=np.int8)
    w_scale = np.zeros((cout,), dtype=np.float32)
    for c in range(cout):
        a = float(np.max(np.abs(w[c])))
        if a < 1e-12:
            a = 1.0
        s = np.float32(a / INT8_LEVELS)
        w_scale[c] = s
        w_int[c] = np.rint(w[c] / s).clip(-127, 127).astype(np.int8)
    return w_int, w_scale


def _wcomp_i8(w_int8: np.ndarray):
    """Wcomp[oc] = 128 * sum(w_int8[oc])  (int32; cancels uint8 act offset)."""
    cout = w_int8.shape[0]
    wcomp = np.empty((cout,), dtype=np.int32)
    flat = w_int8.reshape(cout, -1).astype(np.int32)
    sums = flat.sum(axis=1)
    wcomp[:] = 128 * sums
    return wcomp


def _make_fxp_conv_i8_node(x_name, w, b, xs, y_name, pads, strides, group,
                           uniq, new_inits):
    w_int, w_scale = _quantize_weights_i8(w)
    wcomp = _wcomp_i8(w_int)
    wi_name, ws_name, b_new, wc_name = (
        uniq("Wi8"), uniq("Ws8"), uniq("B8"), uniq("Wcomp8")
    )
    new_inits.append(numpy_helper.from_array(w_int, wi_name))
    new_inits.append(numpy_helper.from_array(w_scale, ws_name))
    new_inits.append(numpy_helper.from_array(b.astype(np.float32), b_new))
    new_inits.append(numpy_helper.from_array(wcomp, wc_name))
    return helper.make_node(
        "FxpConvI8",
        inputs=[x_name, wi_name, b_new, ws_name, wc_name],
        outputs=[y_name],
        name=uniq("conv_i8"),
        domain="com.dcvc",
        x_scale=float(xs),
        group=int(group),
        pads=[int(p) for p in pads],
        strides=[int(s) for s in strides],
    )


def _make_conv_node_dispatch(x_name, w, b, xs_i8, xs_i16, y_name, pads, strides,
                             group, uniq, new_inits, act_bits=16):
    """group==1 -> FxpConvI8 (int8); group!=1 -> FxpConv (int16 fallback)."""
    if group == 1:
        return _make_fxp_conv_i8_node(
            x_name, w, b, xs_i8, y_name, pads, strides, group, uniq, new_inits
        )
    # Depthwise / grouped conv: int8 op is group==1 only -> keep int16.
    return _make_fxp_conv_node(
        x_name, w, b, xs_i16, y_name, pads, strides, group, uniq, new_inits,
        act_bits,
    )


def convert_model_int8(
    src_path: str,
    dst_path: str,
    net_name: str,
    cin: int,
    h: int,
    w: int,
    samples: int,
    calib_inputs: list | None = None,
    percentile: float = 100.0,
):
    m = onnx.load(src_path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    patterns = _find_wsrelu_patterns(g)
    convs = [n for n in g.node if n.op_type == "Conv"]
    graph_outputs = {o.name for o in g.output}
    split_groups = OUTPUT_SPLITS.get(net_name)

    need = set()
    for n in convs:
        need.add(n.input[0])
    for _, _, _, src, _ in patterns:
        need.add(src)
    need.add(g.input[0].name)

    if calib_inputs is None:
        rng = np.random.default_rng(0)
        calib_inputs = [
            {_SINGLE: rng.standard_normal((1, cin, h, w), dtype=np.float32)}
            for _ in range(samples)
        ]
        calib_tag = f"random N(0,1) x{samples}"
    else:
        calib_tag = f"real dumps x{len(calib_inputs)}"
    print(
        f"  calib {len(need)} tensors, {len(convs)} Conv, {len(patterns)} WsRelu "
        f"({calib_tag}, pct={percentile}) ..."
    )
    absmax = _collect_act_ranges(src_path, sorted(need), calib_inputs, percentile)

    remove = set()
    new_inits = []
    skip_names = set()
    for mul4, sig, mulx, _, _ in patterns:
        skip_names.update([mul4.name, sig.name, mulx.name])

    uid = 0

    def uniq(prefix):
        nonlocal uid
        uid += 1
        return f"i8_{prefix}_{uid}"

    n_i8 = 0
    n_i16 = 0
    n_split = 0
    rebuilt = []
    for n in list(g.node):
        if n.name in skip_names:
            pat = None
            for p in patterns:
                if p[2].name == n.name:
                    pat = p
                    break
            if pat is None:
                continue
            mul4, sig, mulx, src, out = pat
            xs = absmax[src] / INT16_LEVELS  # WsRelu LUT is always int16
            lut = _bake_wsrelu_lut(xs)
            lut_name = uniq("ws_lut")
            new_inits.append(numpy_helper.from_array(lut, lut_name))
            rebuilt.append(
                helper.make_node(
                    "FxpWsRelu",
                    inputs=[src, lut_name],
                    outputs=[out],
                    name=uniq("wsrelu"),
                    domain="com.dcvc",
                    x_scale=float(xs),
                )
            )
            continue

        if n.op_type == "Conv":
            attrs = _attr_map(n)
            x_name = n.input[0]
            w_name = n.input[1]
            b_name = n.input[2] if len(n.input) > 2 else None
            y_name = n.output[0]
            w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
            if b_name and b_name in inits:
                b = numpy_helper.to_array(inits[b_name]).astype(np.float32)
            else:
                b = np.zeros((w.shape[0],), np.float32)
            xs_i8 = absmax[x_name] / INT8_LEVELS
            xs_i16 = absmax[x_name] / INT16_LEVELS

            pads = list(attrs.get("pads", [0, 0, 0, 0]))
            strides = list(attrs.get("strides", [1, 1]))
            group = int(attrs.get("group", 1))
            if isinstance(pads[0], bytes):
                pads = [0, 0, 0, 0]
            pads = [int(p) for p in pads]
            strides = [int(s) for s in strides]

            remove.add(w_name)
            if b_name:
                remove.add(b_name)

            do_split = (
                split_groups is not None
                and y_name in graph_outputs
                and group == 1
                and w.shape[2] == 1 and w.shape[3] == 1
                and split_groups[-1][1] == w.shape[0]
            )
            if do_split:
                parts = []
                for lo, hi in split_groups:
                    part_y = uniq(f"split_{lo}_{hi}")
                    parts.append(part_y)
                    rebuilt.append(
                        _make_conv_node_dispatch(
                            x_name, w[lo:hi], b[lo:hi], xs_i8, xs_i16, part_y,
                            pads, strides, group, uniq, new_inits,
                        )
                    )
                    n_i8 += 1
                rebuilt.append(
                    helper.make_node(
                        "Concat",
                        inputs=parts,
                        outputs=[y_name],
                        name=uniq("concat"),
                        axis=1,
                    )
                )
                n_split += 1
            else:
                rebuilt.append(
                    _make_conv_node_dispatch(
                        x_name, w, b, xs_i8, xs_i16, y_name,
                        pads, strides, group, uniq, new_inits,
                    )
                )
                if group == 1:
                    n_i8 += 1
                else:
                    n_i16 += 1
            continue

        rebuilt.append(n)

    keep_inits = [init for init in g.initializer if init.name not in remove]
    keep_inits.extend(new_inits)
    del g.node[:]
    g.node.extend(rebuilt)
    del g.initializer[:]
    g.initializer.extend(keep_inits)

    referenced = set()
    for n in g.node:
        referenced.update(n.input)
    referenced.update(i.name for i in g.input)
    pruned = [i for i in g.initializer if i.name in referenced]
    del g.initializer[:]
    g.initializer.extend(pruned)

    domains = {o.domain for o in m.opset_import}
    if "com.dcvc" not in domains:
        m.opset_import.append(helper.make_opsetid("com.dcvc", 1))

    onnx.save(m, dst_path)
    return {
        "convs": len(convs),
        "conv_i8": n_i8,
        "conv_i16_dw": n_i16,
        "wsrelu": len(patterns),
        "nodes": len(rebuilt),
        "output_splits": n_split,
        "percentile": percentile,
        "n_calib": len(calib_inputs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.path.join(ONNX_DIR, "models_fp32"))
    ap.add_argument("--out-dir", default=os.path.join(ONNX_DIR, "models_int8"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_int8"))
    ap.add_argument("--percentile", type=float, default=100.0)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Copy non-converted assets so the dir is a drop-in model pack.
    for name in os.listdir(args.src_dir):
        s = os.path.join(args.src_dir, name)
        d = os.path.join(args.out_dir, name)
        if os.path.isfile(s) and not name.endswith(".onnx"):
            shutil.copy2(s, d)
        elif os.path.isfile(s) and name.endswith(".onnx"):
            stem = name[:-5]
            if stem not in ENTROPY_NETS and not os.path.exists(d):
                shutil.copy2(s, d)

    nets = args.only or ENTROPY_NETS
    for name in nets:
        src = os.path.join(args.src_dir, f"{name}.onnx")
        dst = os.path.join(args.out_dir, f"{name}.onnx")
        if not os.path.isfile(src):
            print(f"SKIP missing {src}")
            continue
        h, w = NET_HW[name]
        if args.h is not None and name != "hyper_dec":
            h = args.h
        if args.w is not None and name != "hyper_dec":
            w = args.w
        if name == "hyper_dec":
            if args.h is not None:
                h = max(1, args.h // 4)
            if args.w is not None:
                w = max(1, args.w // 4)
        calib_inputs = None
        if args.calib_dir:
            try:
                calib_inputs = _load_calib_inputs(args.calib_dir, name)
            except FileNotFoundError:
                print(f"  [warn] no calib dumps for {name}; using random calib")
                calib_inputs = None
        print(f"=== {name}  cin={NET_CIN[name]}  {h}x{w} ===")
        info = convert_model_int8(
            src, dst, name, NET_CIN[name], h, w, args.samples,
            calib_inputs=calib_inputs, percentile=args.percentile,
        )
        print(f"  wrote {dst}  {info}")

    print("done.")


if __name__ == "__main__":
    main()
