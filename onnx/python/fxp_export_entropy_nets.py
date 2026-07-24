#!/usr/bin/env python3
"""Convert the 7 entropy-parameter ONNX nets to com.dcvc fixed-point ops.

Replaces:
  Conv                         -> com.dcvc::FxpConv  (int16 act / int16 w)
  Mul(x,4)->Sigmoid->Mul(s,x)  -> com.dcvc::FxpWsRelu (65536-entry LUT)

For y_prior_fusion / y_spatial_prior, the final 1x1 that mixes heterogeneous
output channels is split into separate FxpConv groups + Concat:
  prior_fusion:  [qenc|qdec] | [scales] | [means]
  spatial_prior: [scales] | [means]

Leaves Add / Split / DepthToSpace as standard ops (bit-exact on IEEE floats).

Usage:
    uv run python python/fxp_export_entropy_nets.py
    uv run python python/fxp_export_entropy_nets.py --calib-dir python/calib_data_rd \\
        --percentile 99.99 --cle --out-dir models_fxp_real
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

ENTROPY_NETS = [
    # --- INTRA entropy nets ---
    "hyper_dec",
    "y_prior_fusion",
    "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1",
    "y_spatial_prior_adaptor_2",
    "y_spatial_prior_adaptor_3",
    "y_spatial_prior",
    # --- INTER entropy nets (added: fix cross-platform FP32 divergence) ---
    "inter_hyper_dec",
    "inter_temporal_prior",
    "inter_prior_fusion",
    "inter_spatial_prior",
    # --- INTER reconstruction/feature nets (fix FP32 reference drift) ---
    "inter_feature_adaptor_i",
    "inter_hyper_enc",
    "inter_feature_extractor",
    "inter_encoder",
    "inter_decoder",
    "recon_generation",
]

NET_HW = {
    "hyper_dec": (4, 4),
    "y_prior_fusion": (16, 16),
    "y_spatial_prior_reduction": (16, 16),
    "y_spatial_prior_adaptor_1": (16, 16),
    "y_spatial_prior_adaptor_2": (16, 16),
    "y_spatial_prior_adaptor_3": (16, 16),
    "y_spatial_prior": (16, 16),
    "inter_hyper_dec": (4, 4),
    "inter_temporal_prior": (16, 16),
    "inter_prior_fusion": (16, 16),
    "inter_spatial_prior": (16, 16),
    "inter_feature_adaptor_i": (32, 32),
    "inter_hyper_enc": (16, 16),
    "inter_feature_extractor": (32, 32),
    "inter_encoder": (32, 32),
    "inter_decoder": (16, 16),
    "recon_generation": (32, 32),
}

NET_CIN = {
    "hyper_dec": 128,
    "y_prior_fusion": 256,
    "y_spatial_prior_reduction": 514,
    "y_spatial_prior_adaptor_1": 512,
    "y_spatial_prior_adaptor_2": 512,
    "y_spatial_prior_adaptor_3": 512,
    "y_spatial_prior": 512,
    "inter_hyper_dec": 128,
    "inter_temporal_prior": 256,
    "inter_prior_fusion": 384,
    "inter_spatial_prior": 512,
    "inter_feature_adaptor_i": 192,
    "inter_hyper_enc": 128,
    "inter_feature_extractor": 256,
    "inter_encoder": 192,
    "inter_decoder": 128,
    "recon_generation": 256,
}

# Final-output channel groups (half-open ranges). Only applied when the Conv
# writes a graph output and Cout matches the span.
OUTPUT_SPLITS = {
    # params_fusion layout: ch0 qenc, ch1 qdec, ch2:258 scales, ch258:514 means
    "y_prior_fusion": [(0, 2), (2, 258), (258, 514)],
    # spatial prior: ch0:256 scales, ch256:512 means
    "y_spatial_prior": [(0, 256), (256, 512)],
    # inter prior_fusion: ch0:128 q_dec, ch128:256 scales, ch256:384 means
    "inter_prior_fusion": [(0, 128), (128, 256), (256, 384)],
    # inter spatial prior: ch0:128 scales1, ch128:256 means1
    "inter_spatial_prior": [(0, 128), (128, 256)],
}


def _attr_map(node):
    out = {}
    for a in node.attribute:
        out[a.name] = helper.get_attribute_value(a)
    return out


def _find_wsrelu_patterns(graph):
    """Return list of (mul4, sigmoid, mulx, src_tensor, out_tensor)."""
    by_out = {n.output[0]: n for n in graph.node}
    inits = {i.name: i for i in graph.initializer}
    patterns = []
    for n in graph.node:
        if n.op_type != "Mul" or len(n.input) != 2:
            continue
        # Second Mul of pattern: Mul(sigmoid, src)
        a, b = n.input
        sig_node = by_out.get(a)
        src = b
        if sig_node is None or sig_node.op_type != "Sigmoid":
            sig_node = by_out.get(b)
            src = a
        if sig_node is None or sig_node.op_type != "Sigmoid":
            continue
        mul4 = by_out.get(sig_node.input[0])
        if mul4 is None or mul4.op_type != "Mul":
            continue
        # mul4 should be Mul(src, 4) or Mul(4, src)
        m0, m1 = mul4.input
        const_name = None
        if m0 == src:
            const_name = m1
        elif m1 == src:
            const_name = m0
        else:
            continue
        if const_name not in inits:
            continue
        c = float(numpy_helper.to_array(inits[const_name]).reshape(-1)[0])
        if abs(c - 4.0) > 1e-5:
            continue
        patterns.append((mul4, sig_node, n, src, n.output[0]))
    return patterns


def _quantize_weights(w: np.ndarray):
    """Per-output-channel symmetric int16 quantization."""
    cout = w.shape[0]
    w_int = np.zeros_like(w, dtype=np.int16)
    w_scale = np.zeros((cout,), dtype=np.float32)
    for c in range(cout):
        a = float(np.max(np.abs(w[c])))
        if a < 1e-12:
            a = 1.0
        s = np.float32(a / 32767.0)
        w_scale[c] = s
        w_int[c] = np.rint(w[c] / s).clip(-32767, 32767).astype(np.int16)
    return w_int, w_scale


def _make_fxp_conv_node(x_name, w, b, xs, y_name, pads, strides, group, uniq, new_inits, act_bits=16):
    w_int, w_scale = _quantize_weights(w)
    wi_name, ws_name, b_new = uniq("W"), uniq("Ws"), uniq("B")
    new_inits.append(numpy_helper.from_array(w_int, wi_name))
    new_inits.append(numpy_helper.from_array(w_scale, ws_name))
    new_inits.append(numpy_helper.from_array(b.astype(np.float32), b_new))
    node = helper.make_node(
        "FxpConv",
        inputs=[x_name, wi_name, b_new, ws_name],
        outputs=[y_name],
        name=uniq("conv"),
        domain="com.dcvc",
        x_scale=float(xs),
        group=int(group),
        pads=[int(p) for p in pads],
        strides=[int(s) for s in strides],
    )
    if act_bits == 32:
        node.attribute.append(helper.make_attribute("act_bits", 32))
    return node


def _make_det_conv_node(x_name, w, b, y_name, pads, strides, group, uniq, new_inits):
    """DetConv: float32 weights, float64 accumulation. No quantization."""
    w_name, b_new = uniq("W_det"), uniq("B_det")
    new_inits.append(numpy_helper.from_array(w.astype(np.float32), w_name))
    new_inits.append(numpy_helper.from_array(b.astype(np.float32), b_new))
    return helper.make_node(
        "DetConv",
        inputs=[x_name, w_name, b_new],
        outputs=[y_name],
        name=uniq("detconv"),
        domain="com.dcvc",
        group=int(group),
        pads=[int(p) for p in pads],
        strides=[int(s) for s in strides],
    )


def _bake_wsrelu_lut(x_scale: float) -> np.ndarray:
    q = np.arange(-32768, 32768, dtype=np.float64)
    x = q * float(x_scale)
    # Stable sigmoid
    lut = x * (1.0 / (1.0 + np.exp(-np.clip(4.0 * x, -60, 60))))
    return lut.astype(np.float32)


_SINGLE = "_single_"  # sentinel key for single-input .npy calib samples


def _load_calib_inputs(calib_dir: str, net_name: str):
    """Load dumped net inputs. Returns list of feed-dicts (one per sample).

    Supports two layouts:
      .npy  -> single-input net; key is _SINGLE (resolved to graph input[0]
               by the caller). 4-D normalisation is applied.
      .npz  -> multi-input net; keys are ONNX input tensor names.
    Accepts both ``sample_N`` and legacy ``N`` naming.
    """
    import glob

    paths = sorted(glob.glob(os.path.join(calib_dir, net_name, "sample_*")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(calib_dir, net_name, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"no calib dumps in {calib_dir}/{net_name}")
    samples = []
    for p in paths:
        if p.endswith(".npz"):
            z = np.load(p)
            samples.append({k: z[k].astype(np.float32) for k in z.files})
        else:
            a = np.load(p).astype(np.float32)
            if a.ndim == 3:
                a = a[None]
            samples.append({_SINGLE: a})
    return samples


def _collect_act_ranges(
    model_path: str,
    tensor_names,
    inputs: list,
    percentile: float,
):
    """Run FP32 model with extra outputs; return {name: abs_range}.

    percentile=100 → global absmax; otherwise global |x| percentile across all
    calib samples (then never below a tiny epsilon).
    """
    m = onnx.load(model_path)
    graph_in = m.graph.input[0].name
    existing = {o.name for o in m.graph.output}
    for name in tensor_names:
        if name not in existing and name != graph_in:
            m.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
            existing.add(name)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    sess_inputs = sess.get_inputs()
    in_names = [i.name for i in sess_inputs]
    out_names = [o.name for o in sess.get_outputs()]
    buckets = {t: [] for t in tensor_names}
    for feeds in inputs:
        feed = {}
        for si in sess_inputs:
            if si.name in feeds:
                feed[si.name] = feeds[si.name]
            elif _SINGLE in feeds:
                feed[si.name] = feeds[_SINGLE]
            else:
                feed[si.name] = np.zeros(
                    [d if isinstance(d, int) else 1 for d in si.shape],
                    dtype=np.float32,
                )
        outs = sess.run(None, feed)
        by = dict(zip(out_names, outs))
        for nm in in_names:
            if nm in buckets:
                buckets[nm].append(np.abs(feed[nm]).ravel())
        for t in tensor_names:
            if t in by:
                buckets[t].append(np.abs(by[t]).ravel())

    ranges = {}
    for t, parts in buckets.items():
        if not parts:
            ranges[t] = 1.0
            continue
        v = np.concatenate(parts)
        if v.size == 0:
            ranges[t] = 1.0
            continue
        if percentile >= 100.0:
            r = float(np.max(v))
        else:
            r = float(np.percentile(v, percentile))
        ranges[t] = r if r >= 1e-8 else 1.0
    return ranges


def convert_model(
    src_path: str,
    dst_path: str,
    net_name: str,
    cin: int,
    h: int,
    w: int,
    samples: int,
    calib_inputs: list | None = None,
    percentile: float = 100.0,
    act_bits: int = 16,
    det_conv: bool = False,
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
        f"({calib_tag}, pct={percentile}) …"
    )
    absmax = _collect_act_ranges(src_path, sorted(need), calib_inputs, percentile)

    remove = set()
    new_inits = []

    wsrelu_by_mul4 = {p[0].name: p for p in patterns}
    skip_names = set()
    for mul4, sig, mulx, _, _ in patterns:
        skip_names.update([mul4.name, sig.name, mulx.name])

    act_levels = 2147483647.0 if act_bits == 32 else 32767.0

    def scale_of(tensor):
        return absmax[tensor] / act_levels

    uid = 0

    def uniq(prefix):
        nonlocal uid
        uid += 1
        return f"fxp_{prefix}_{uid}"

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
            xs = absmax[src] / 32767.0  # WsRelu LUT is always int16 (65536 entries)
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
            if det_conv:
                x_name = n.input[0]
                w_name = n.input[1]
                b_name = n.input[2] if len(n.input) > 2 else None
                y_name = n.output[0]
                w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
                if b_name and b_name in inits:
                    b = numpy_helper.to_array(inits[b_name]).astype(np.float32)
                else:
                    b = np.zeros((w.shape[0],), np.float32)
                remove.add(w_name)
                if b_name:
                    remove.add(b_name)
                pads = list(attrs.get("pads", [0, 0, 0, 0]))
                strides = list(attrs.get("strides", [1, 1]))
                group = int(attrs.get("group", 1))
                if isinstance(pads[0], bytes):
                    pads = [0, 0, 0, 0]
                rebuilt.append(
                    _make_det_conv_node(
                        x_name, w, b, y_name, pads, strides, group, uniq, new_inits,
                    )
                )
                continue
            x_name = n.input[0]
            w_name = n.input[1]
            b_name = n.input[2] if len(n.input) > 2 else None
            y_name = n.output[0]
            w = numpy_helper.to_array(inits[w_name]).astype(np.float32)
            if b_name and b_name in inits:
                b = numpy_helper.to_array(inits[b_name]).astype(np.float32)
            else:
                b = np.zeros((w.shape[0],), np.float32)
            xs = scale_of(x_name)

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
                        _make_fxp_conv_node(
                            x_name, w[lo:hi], b[lo:hi], xs, part_y,
                            pads, strides, group, uniq, new_inits, act_bits,
                        )
                    )
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
                    _make_fxp_conv_node(
                        x_name, w, b, xs, y_name, pads, strides, group, uniq, new_inits, act_bits,
                    )
                )
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
        "wsrelu": len(patterns),
        "nodes": len(rebuilt),
        "output_splits": n_split,
        "weight_bits": 16,
        "percentile": percentile,
        "n_calib": len(calib_inputs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=os.path.join(ONNX_DIR, "models"))
    ap.add_argument("--out-dir", default=os.path.join(ONNX_DIR, "models_fxp"))
    ap.add_argument("--samples", type=int, default=8,
                    help="random calib count when --calib-dir is not set")
    ap.add_argument("--calib-dir", default=None,
                    help="ptq_dump_calib layout: <dir>/<net>/*.npy (preferred)")
    ap.add_argument("--percentile", type=float, default=100.0,
                    help="activation |x| percentile for x_scale (100=absmax)")
    ap.add_argument("--cle", action="store_true",
                    help="apply CLE on FP32 weights before FXP conversion")
    ap.add_argument("--h", type=int, default=None, help="override H for y-plane nets")
    ap.add_argument("--w", type=int, default=None, help="override W for y-plane nets")
    ap.add_argument("--only", nargs="*", default=None, help="subset of net names")
    ap.add_argument("--act-bits", type=int, default=16, choices=[16, 32],
                    help="activation quantization bits (16=default, 32=near-lossless)")
    ap.add_argument("--det-conv", action="store_true",
                    help="use DetConv (float32 weights, float64 acc) instead of FxpConv — "
                         "eliminates weight quantization loss for reconstruction nets")
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

    cle_tmpdir = None
    src_dir = args.src_dir
    if args.cle:
        from cle_equalize import cle_equalize_model

        cle_tmpdir = os.path.join(args.out_dir, "_cle_tmp")
        os.makedirs(cle_tmpdir, exist_ok=True)
        for name in (args.only or ENTROPY_NETS):
            src = os.path.join(args.src_dir, f"{name}.onnx")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(cle_tmpdir, f"{name}.onnx")
            n = cle_equalize_model(src, dst, iterations=2)
            print(f"CLE {name}: pairs={n}")
        src_dir = cle_tmpdir

    nets = args.only or ENTROPY_NETS
    for name in nets:
        src = os.path.join(src_dir, f"{name}.onnx")
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
            # z spatial = y/4; if y overridden, scale z
            if args.h is not None:
                h = max(1, args.h // 4)
            if args.w is not None:
                w = max(1, args.w // 4)
        calib_inputs = None
        if args.calib_dir:
            calib_inputs = _load_calib_inputs(args.calib_dir, name)
        print(f"=== {name}  cin={NET_CIN[name]}  {h}x{w} ===")
        info = convert_model(
            src, dst, name, NET_CIN[name], h, w, args.samples,
            calib_inputs=calib_inputs, percentile=args.percentile,
            act_bits=args.act_bits, det_conv=args.det_conv,
        )
        print(f"  wrote {dst}  {info}")

    if cle_tmpdir and os.path.isdir(cle_tmpdir):
        shutil.rmtree(cle_tmpdir, ignore_errors=True)

    print("done.")


if __name__ == "__main__":
    main()
