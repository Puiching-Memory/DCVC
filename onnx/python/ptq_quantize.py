#!/usr/bin/env python3
"""INT8 static PTQ of the DCVC-UF intra entropy-parameter nets.

Quantizes the 7 networks that produce the rANS entropy parameters
(hyper_dec, y_prior_fusion, y_spatial_prior_reduction,
y_spatial_prior_adaptor_{1,2,3}, y_spatial_prior) with
onnxruntime.quantization.quantize_static:

    activation u8, weight s8, per_channel=True,
    op_types_to_quantize=['Conv', 'MatMul', 'Gemm']

Only Conv/MatMul/Gemm are quantized; all elementwise ops (Sigmoid, Mul, Add,
DepthToSpace, ...) stay FP32. Integer conv accumulation is order-independent,
so the quantized nets are bit-exact across CPU platforms / thread counts /
graph optimization levels, which keeps the rANS scale indexes in sync.

--format qoperator (default) emits QLinearConv nodes directly, so the integer
kernels run even with ORT_DISABLE_ALL and outputs are bit-identical under all
four thread x graph-opt configurations. --format qdq only gets integer kernels
after ORT's graph-optimization fusion (the C side runs the default
ORT_ENABLE_ALL, so QDQ is bit-exact in deployment). QDQ is required for
int16 activations (QLinearConv only accepts int8/uint8 in its type schema).
QDQ with int8 activations fails the strict 4-config bit-exact test because
ORT_DISABLE_ALL executes FP32 convs on the dequantized values, but int16
QDQ is verified bit-exact under ORT 1.27's default ORT_ENABLE_ALL.

Produces onnx/models_int8/: a full copy of onnx/models/ with the 7 onnx files
replaced by their quantized versions. The IR version is pinned back to the
source value (8) so the models load with the ORT 1.27 CPU EP used by the C side.

After quantization each model is checked with onnx: every Conv's weight input
must be produced by a DequantizeLinear, resp. every conv must be a QLinearConv
(no bare FP32 Conv left); per-model Q/DQ node counts and residual FP32 Conv
counts are printed.

Usage:
  python ptq_quantize.py [--format qoperator|qdq] [--models-dir ../models]
                         [--calib-dir calib_data] [--out-dir ../models_int8]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

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

SOURCE_IR_VERSION = 8  # matches onnx/models/*.onnx; must stay <= what ORT 1.27 accepts

# Optional FP32-equivalent output-conv channel-group split (applied before
# quantization). The final 1x1 Conv of these nets emits channels with very
# different ranges (qenc/qdec pre-sigmoid, scales, means). Per-tensor u8
# activation quantization of the fused output is dominated by the means range
# (+-20), leaving the scales (order 0.1..3) only a few quantization levels and
# wrecking RD. Splitting the conv into per-group convs (functionally
# bit-identical in FP32: each output channel's dot product is unchanged) lets
# each group get its own activation quantization range.
SPLIT_OUTPUT_GROUPS = {
    "y_prior_fusion": [(0, 2), (2, 258), (258, 514)],
    "y_spatial_prior": [(0, 256), (256, 512)],
}


def split_output_conv(model_path, groups, work_path):
    """Split the Conv producing the graph output into per-group Convs + Concat.

    FP32-equivalent (verified numerically by the caller). Returns work_path.
    """
    import onnx.numpy_helper as nh

    model = onnx.load(model_path)
    graph = model.graph
    out_name = graph.output[0].name
    producer = {o: n for n in graph.node for o in n.output}
    conv = producer[out_name]
    assert conv.op_type == "Conv", f"output producer is {conv.op_type}, expected Conv"
    inits = {i.name: i for i in graph.initializer}
    w = nh.to_array(inits[conv.input[1]])
    b = nh.to_array(inits[conv.input[2]]) if len(conv.input) > 2 else None
    assert w.shape[0] == groups[-1][1]

    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
    new_nodes = []
    concat_inputs = []
    for gi, (c0, c1) in enumerate(groups):
        wn = f"{conv.input[1]}_g{gi}"
        graph.initializer.append(nh.from_array(np.ascontiguousarray(w[c0:c1]), wn))
        inputs = [conv.input[0], wn]
        if b is not None:
            bn = f"{conv.input[2]}_g{gi}"
            graph.initializer.append(nh.from_array(np.ascontiguousarray(b[c0:c1]), bn))
            inputs.append(bn)
        o = f"{out_name}_g{gi}"
        new_nodes.append(onnx.helper.make_node("Conv", inputs, [o],
                                               name=f"{conv.name}_g{gi}", **attrs))
        concat_inputs.append(o)
    new_nodes.append(onnx.helper.make_node("Concat", concat_inputs, [out_name],
                                           name=f"{conv.name}_concat", axis=1))
    graph.node.remove(conv)
    graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, work_path)
    return work_path


class NpyCalibrationReader(CalibrationDataReader):
    """Feeds sorted .npy files of one directory as {'in0': array}."""

    def __init__(self, npy_dir, input_name="in0"):
        self.files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"no npy files in {npy_dir}")
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        arr = np.load(self.files[self.idx]).astype(np.float32)
        self.idx += 1
        return {self.input_name: arr}

    def rewind(self):
        self.idx = 0

    def __len__(self):
        return len(self.files)


def check_quantized_model(path):
    """Return stats incl. residual FP32 Convs (weight not from DQ / not QLinearConv)."""
    model = onnx.load(path, load_external_data=False)
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
    n_q = n_dq = n_conv = n_fp32_conv = n_qconv = 0
    fp32_convs = []
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear":
            n_q += 1
        elif node.op_type == "DequantizeLinear":
            n_dq += 1
        elif node.op_type == "QLinearConv":
            n_qconv += 1
        elif node.op_type == "Conv":
            n_conv += 1
            w_prod = producer.get(node.input[1])
            if w_prod is None or w_prod.op_type != "DequantizeLinear":
                n_fp32_conv += 1
                fp32_convs.append(node.name or node.output[0])
    return {
        "ir_version": model.ir_version,
        "opset": max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        "quantize_linear": n_q,
        "dequantize_linear": n_dq,
        "conv": n_conv,
        "qlinear_conv": n_qconv,
        "fp32_conv": n_fp32_conv,
        "fp32_conv_names": fp32_convs,
    }


def quantize_one(name, models_dir, calib_dir, out_dir, quant_format, calibrate_method,
                 split_output=True, activation_type=QuantType.QUInt8,
                 use_cle=False, percentile=99.999):
    src = os.path.join(models_dir, name + ".onnx")
    dst = os.path.join(out_dir, name + ".onnx")
    reader = NpyCalibrationReader(os.path.join(calib_dir, name))
    quant_input = src
    cle_tmp = None
    if use_cle:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(__file__))
        from cle_equalize import cle_equalize_model
        cle_tmp = os.path.join(out_dir, f".{name}_cle_tmp.onnx")
        cle_equalize_model(src, cle_tmp, iterations=2)
        quant_input = cle_tmp
        src = cle_tmp  # downstream split reads from the CLE'd model
    if split_output and name in SPLIT_OUTPUT_GROUPS:
        work = os.path.join(out_dir, f".{name}_split_tmp.onnx")
        split_output_conv(src, SPLIT_OUTPUT_GROUPS[name], work)
        # FP32-equivalence check: split must not change outputs bit-for-bit.
        feed = reader.get_next()
        reader.rewind()
        o_ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        o_new = ort.InferenceSession(work, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        if o_ref.tobytes() != o_new.tobytes():
            md = float(np.abs(o_ref - o_new).max())
            raise RuntimeError(f"{name}: split not FP32-equivalent (maxdiff={md})")
        quant_input = work
    extra_opts = {}
    if calibrate_method == CalibrationMethod.Percentile:
        extra_opts["CalibPercentile"] = percentile
    quantize_static(
        model_input=quant_input,
        model_output=dst,
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        calibrate_method=calibrate_method,
        extra_options=extra_opts if extra_opts else None,
    )
    if quant_input != src and os.path.exists(quant_input):
        os.remove(quant_input)
    if cle_tmp and os.path.exists(cle_tmp):
        os.remove(cle_tmp)
    # Pin IR version back to the source value for ORT 1.27 compatibility.
    model = onnx.load(dst, load_external_data=False)
    if model.ir_version != SOURCE_IR_VERSION:
        model.ir_version = SOURCE_IR_VERSION
        onnx.save(model, dst)
    stats = check_quantized_model(dst)

    # Sanity: run one calibration input through the quantized model.
    reader.rewind()
    feed = reader.get_next()
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    out = sess.run(None, feed)[0]
    stats["out_shape"] = list(out.shape)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.normpath(os.path.join(REPO, "..", "models")))
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_data"))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_int8")))
    ap.add_argument("--format", choices=["qoperator", "qdq"], default="qoperator",
                    help="qoperator (default): QLinearConv nodes in the graph, integer "
                         "kernels even with ORT_DISABLE_ALL -> bit-identical outputs "
                         "across all thread x graph-opt configs. qdq: Q/DQ wrappers; "
                         "ORT only fuses them into integer kernels when graph "
                         "optimization is enabled, so ORT_DISABLE_ALL runs FP32 convs "
                         "and outputs differ (fails the 4-config bit-exact test).")
    ap.add_argument("--activation-type", choices=["uint8", "int16"], default="uint8",
                    help="activation quantization type. int16 forces --format qdq "
                         "(QLinearConv only accepts int8/uint8) and bumps the model "
                         "opset to 21 (native int16 Q/DQ in ONNX opset 21); weights "
                         "stay s8 per-channel. 16-bit activations give a 256x finer "
                         "grid than u8, dramatically reducing quantization error on "
                         "these multi-range entropy-parameter activations (measured: "
                         "+3.4% bitrate overhead vs +67% for best u8 config). "
                         "Requires ORT >= 1.20 (the C side uses ORT 1.27).")
    ap.add_argument("--calibrate-method", choices=["minmax", "entropy", "percentile", "distribution"],
                    default="minmax", help="activation range calibration method")
    ap.add_argument("--percentile", type=float, default=99.999,
                    help="percentile value for percentile calibration (default 99.999)")
    ap.add_argument("--cle", action="store_true",
                    help="apply Cross-Layer Equalization (data-free weight transform) "
                         "before quantization to equalize per-channel weight ranges")
    ap.add_argument("--no-split", action="store_true",
                    help="disable the FP32-equivalent output-conv channel-group split "
                         "for y_prior_fusion / y_spatial_prior")
    args = ap.parse_args()
    activation_type = QuantType.QUInt8 if args.activation_type == "uint8" else QuantType.QInt16
    if args.activation_type == "int16" and args.format != "qdq":
        print("note: int16 activations require QDQ; overriding --format to qdq")
    quant_format = (QuantFormat.QOperator if args.format == "qoperator" else QuantFormat.QDQ)
    if activation_type == QuantType.QInt16:
        quant_format = QuantFormat.QDQ
    calibrate_method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }[args.calibrate_method]

    # Full copy of the model dir (npy entropy tables, q tables, untouched nets).
    if os.path.abspath(args.models_dir) == os.path.abspath(args.out_dir):
        print("out dir must differ from models dir", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.models_dir):
        s = os.path.join(args.models_dir, f)
        d = os.path.join(args.out_dir, f)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)

    print(f"format={'qdq' if quant_format == QuantFormat.QDQ else 'qoperator'} "
          f"activation={args.activation_type} calibrate_method={args.calibrate_method} "
          f"split_output={not args.no_split} cle={args.cle} percentile={args.percentile}")
    print(f"{'model':36s} {'Q':>4s} {'DQ':>4s} {'QLConv':>7s} {'Conv':>5s} {'fp32Conv':>8s} "
          f"{'ir':>3s} {'opset':>5s}  out_shape")
    all_ok = True
    for name in TARGET_NETS:
        stats = quantize_one(name, args.models_dir, args.calib_dir, args.out_dir,
                             quant_format, calibrate_method, split_output=not args.no_split,
                             activation_type=activation_type,
                             use_cle=args.cle, percentile=args.percentile)
        print(f"{name:36s} {stats['quantize_linear']:4d} {stats['dequantize_linear']:4d} "
              f"{stats['qlinear_conv']:7d} {stats['conv']:5d} {stats['fp32_conv']:8d} "
              f"{stats['ir_version']:3d} {stats['opset']:5d}  {stats['out_shape']}")
        if stats["fp32_conv"]:
            all_ok = False
            print(f"  WARNING: residual FP32 Convs: {stats['fp32_conv_names']}")
    print("ALL CONVS QUANTIZED" if all_ok else "RESIDUAL FP32 CONVS FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
