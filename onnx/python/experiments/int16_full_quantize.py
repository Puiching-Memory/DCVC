#!/usr/bin/env python3
"""Pure-ONNX (no FXP) full-model INT16 quantization of the DCVC intra codec.

Quantizes ALL 10 intra subnets used by the C end-to-end pipeline
(test_cpu_end2end round-trip) to INT16 QDQ using onnxruntime.quantization:

    activation QInt16, weight QInt16, per-channel, format=QDQ,
    op_types_to_quantize=['Conv','MatMul','Gemm']

Base model dir is onnx/models_fp32 (pure standard-Conv ONNX, no FxpConv/FxpWsRelu).
Output is a full model-dir copy onnx/models_int16 with the 10 onnx replaced.

Calibration data is generated on the fly by reproducing the C encoder-side
dataflow (reused from ptq_dump_calib.Encoder) and dumping every subnet's real
inputs across several frames x qps. For the two-input nets
(intra_analysis_standard, intra_synthesis) paired in0/in1 samples are saved.

Usage:
  .venv/bin/python onnx/python/int16_full_quantize.py \
      [--weight-type int16|int8] [--frames beauty bosphorus jockey] \
      [--qps 12 32 52] [--crop 256] [--skip-calib]
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
sys.path.insert(0, REPO)
from ptq_dump_calib import Encoder, load_frame  # noqa: E402

MODELS_FP32 = os.path.normpath(os.path.join(REPO, "..", "models_fp32"))
DEFAULT_OUT = os.path.normpath(os.path.join(REPO, "..", "models_int16"))
DEFAULT_CALIB = os.path.join(REPO, "calib_int16_full")
DEFAULT_FRAMES = os.path.normpath(os.path.join(REPO, "..", "rd_frames"))

TEST_SEQ_ROOT = os.path.normpath(os.path.join(REPO, "..", "..", "datasets", "test_sequences", "YUV"))

SPLIT_OUTPUT_GROUPS = {
    "y_prior_fusion": [(0, 2), (2, 258), (258, 514)],
    "y_spatial_prior": [(0, 256), (256, 512)],
}


def split_output_conv(model_path, groups, work_path):
    """Split the Conv producing the graph output into per-group Convs + Concat.
    FP32-equivalent (verified numerically by the caller)."""
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


KR_, KG_, KB_ = 0.2126, 0.7152, 0.0722

# (class, name, w, h, frames) - diverse calibration content
CALIB_SEQS = [
    ("UVG", "Beauty_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
    ("UVG", "Jockey_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
    ("UVG", "YachtRide_1920x1080_120fps_420_8bit_YUV", 1920, 1080),
    ("HEVC_E", "Johnny_1280x720_60", 1280, 720),
    ("HEVC_E", "KristenAndSara_1280x720_60", 1280, 720),
]


def read_yuv420_frame(path, w, h, idx):
    ys, cs = w * h, (w // 2) * (h // 2)
    fs = ys + 2 * cs
    with open(path, "rb") as f:
        f.seek(idx * fs)
        d = f.read(fs)
    if len(d) < fs:
        raise EOFError(f"frame {idx} out of range in {path}")
    y = np.frombuffer(d[:ys], np.uint8).reshape(h, w).astype(np.float32) / 255.0
    u = np.frombuffer(d[ys:ys + cs], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    v = np.frombuffer(d[ys + cs:], np.uint8).reshape(h // 2, w // 2).astype(np.float32) / 255.0
    return y, u, v


def yuv420_to_rgb(y, u, v):
    h, w = y.shape
    uu = np.repeat(np.repeat(u, 2, 0), 2, 1)[:h, :w]
    vv = np.repeat(np.repeat(v, 2, 0), 2, 1)[:h, :w]
    cb = (uu - 0.5) * 2.0 * (1.0 - KB_)
    cr = (vv - 0.5) * 2.0 * (1.0 - KR_)
    r = y + cr
    b = y + cb
    g = (y - KR_ * r - KB_ * b) / KG_
    rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)])
    return np.ascontiguousarray(rgb[None].astype(np.float32))


def dump_calibration_yuv(models_dir, calib_seqs, qps, crop, frame_idxs, out_dir):
    """Dump calibration from diverse YUV test sequences + full qp range."""
    from ptq_dump_calib import Encoder, rgb_to_ycbcr_c
    enc = Encoder(models_dir)
    qdec = np.load(os.path.join(models_dir, "q_scale_dec.npy"))
    for net in INTRA_NETS:
        os.makedirs(os.path.join(out_dir, net), exist_ok=True)
    si = 0
    for cls, name, w, h in calib_seqs:
        yuv_path = os.path.join(TEST_SEQ_ROOT, cls, name + ".yuv")
        if not os.path.exists(yuv_path):
            print(f"  skip {name}: not found"); continue
        cw = ch = min(crop, w, h)
        cw = cw // 64 * 64; ch = ch // 64 * 64
        i0, j0 = (h - ch) // 2, (w - cw) // 2
        for fidx in frame_idxs:
            y, u, v = read_yuv420_frame(yuv_path, w, h, fidx)
            yc = y[i0:i0+ch, j0:j0+cw]
            uc = u[i0//2:i0//2+ch//2, j0//2:j0//2+cw//2]
            vc = v[i0//2:i0//2+ch//2, j0//2:j0//2+cw//2]
            x = yuv420_to_rgb(yc, uc, vc)
            for qp in qps:
                stem = f"s{si:02d}_{name.split('_')[0]}_f{fidx}_qp{qp}"
                d = {}
                res = enc.encode_sample(x, qp, lambda n, _, a: d.__setitem__(n, a))
                from ptq_dump_calib import rgb_to_ycbcr_c as _r
                x_ycbcr = _r(x[0])[None]
                qenc = enc.q_scale_enc[qp:qp+1].astype(np.float32)
                np.save(os.path.join(out_dir, "intra_analysis_standard", f"{stem}_in0.npy"), np.ascontiguousarray(x_ycbcr))
                np.save(os.path.join(out_dir, "intra_analysis_standard", f"{stem}_in1.npy"), np.ascontiguousarray(qenc))
                np.save(os.path.join(out_dir, "intra_hyper_enc", f"{stem}.npy"), np.ascontiguousarray(res["y"]))
                np.save(os.path.join(out_dir, "intra_synthesis", f"{stem}_in0.npy"), np.ascontiguousarray(res["y_hat"]))
                np.save(os.path.join(out_dir, "intra_synthesis", f"{stem}_in1.npy"), np.ascontiguousarray(qdec[qp:qp+1].astype(np.float32)))
                for nm, arr in d.items():
                    np.save(os.path.join(out_dir, nm, f"{stem}.npy"), np.ascontiguousarray(arr))
                print(f"  calib {stem}", flush=True); si += 1
    print(f"YUV calibration: {si} samples -> {out_dir}")


# The 10 intra subnets loaded by cpu_intra_pipeline + cpu_ar_codec.
INTRA_NETS = [
    "intra_analysis_standard",
    "intra_hyper_enc",
    "hyper_dec",
    "y_prior_fusion",
    "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1",
    "y_spatial_prior_adaptor_2",
    "y_spatial_prior_adaptor_3",
    "y_spatial_prior",
    "intra_synthesis",
]
TWO_INPUT = {"intra_analysis_standard", "intra_synthesis"}


def dump_calibration(models_dir, frames_dir, frames, qps, crop, out_dir):
    """Run the FP32 encoder dataflow and dump every subnet's inputs."""
    enc = Encoder(models_dir)
    qdec = np.load(os.path.join(models_dir, "q_scale_dec.npy"))  # [64,368,1,1]
    for net in INTRA_NETS:
        os.makedirs(os.path.join(out_dir, net), exist_ok=True)

    si = 0
    for frame in frames:
        x = load_frame(frames_dir, frame, crop)
        _, _, h, w = x.shape
        for qp in qps:
            stem = f"s{si:02d}_{frame}_qp{qp}"
            d = {}

            def dump(name, _idx, arr):
                d[name] = arr

            res = enc.encode_sample(x, qp, dump)
            # analysis: in0=x_ycbcr, in1=qenc  -> capture via a second run with dump hook
            # Encoder.encode_sample already ran analysis internally; recompute inputs:
            # (cheap: just re-run the two input-producing calls)
            # analysis inputs
            from ptq_dump_calib import rgb_to_ycbcr_c
            x_ycbcr = rgb_to_ycbcr_c(x[0])[None]
            qenc = enc.q_scale_enc[qp:qp + 1].astype(np.float32)
            np.save(os.path.join(out_dir, "intra_analysis_standard", f"{stem}_in0.npy"),
                    np.ascontiguousarray(x_ycbcr))
            np.save(os.path.join(out_dir, "intra_analysis_standard", f"{stem}_in1.npy"),
                    np.ascontiguousarray(qenc))
            # hyper_enc input = clipped y
            y = res["y"]
            np.save(os.path.join(out_dir, "intra_hyper_enc", f"{stem}.npy"),
                    np.ascontiguousarray(y))
            # synthesis input = y_hat (decoded latent), in1 = qdec
            yhat = res["y_hat"]
            np.save(os.path.join(out_dir, "intra_synthesis", f"{stem}_in0.npy"),
                    np.ascontiguousarray(yhat))
            np.save(os.path.join(out_dir, "intra_synthesis", f"{stem}_in1.npy"),
                    np.ascontiguousarray(qdec[qp:qp + 1].astype(np.float32)))
            # the 7 entropy nets were already dumped into d by the callback
            for name, arr in d.items():
                np.save(os.path.join(out_dir, name, f"{stem}.npy"), np.ascontiguousarray(arr))
            print(f"  calib {stem}: {frame} qp={qp} {h}x{w}", flush=True)
            si += 1
    print(f"calibration written to {out_dir} ({si} samples)")


class MultiInputCalibReader(CalibrationDataReader):
    """Feeds calibration samples. Single-input nets use {stem}.npy;
    two-input nets use {stem}_in0.npy / {stem}_in1.npy keyed by graph input names."""

    def __init__(self, net_dir, model_path, qp_filter=None):
        m = onnx.load(model_path, load_external_data=False)
        self.inputs = [i.name for i in m.graph.input]
        files = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(net_dir, "*.npy"))]
        stems = set()
        for fname in files:
            stem = fname
            for k in self.inputs:
                if fname.endswith("_" + k):
                    stem = fname[: -(len(k) + 1)]
                    break
            stems.add(stem)
        # When qp_filter is set, keep only calibration samples matching that qp.
        if qp_filter is not None:
            tag = f"qp{qp_filter}"
            stems = {st for st in stems if st.endswith(tag) or f"_{tag}" in st}
        self.stems = sorted(stems)
        self.idx = 0
        self.net_dir = net_dir

    def get_next(self):
        if self.idx >= len(self.stems):
            return None
        stem = self.stems[self.idx]
        self.idx += 1
        feed = {}
        for k in self.inputs:
            p = os.path.join(self.net_dir, f"{stem}_{k}.npy")
            if not os.path.exists(p):  # single-input convention
                p = os.path.join(self.net_dir, f"{stem}.npy")
            feed[k] = np.load(p).astype(np.float32)
        return feed

    def rewind(self):
        self.idx = 0

    def __len__(self):
        return len(self.stems)


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def quantize_one(name, models_dir, calib_dir, out_dir, weight_type, calib_method, qp_filter=None):
    src = os.path.join(models_dir, name + ".onnx")
    dst = os.path.join(out_dir, name + ".onnx")
    net_calib = os.path.join(calib_dir, name)
    # Apply FP32-equivalent output-conv split for entropy nets (each channel
    # group gets its own int16 activation range; without this the means range
    # +-20 dominates and starves the scales ~0.1-3 of quantization levels).
    quant_input = src
    work = None
    if name in SPLIT_OUTPUT_GROUPS:
        work = os.path.join(out_dir, f".{name}_split.onnx")
        split_output_conv(src, SPLIT_OUTPUT_GROUPS[name], work)
        # verify FP32-equivalence
        reader0 = MultiInputCalibReader(net_calib, src, qp_filter)
        feed = reader0.get_next()
        o_ref = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        o_new = ort.InferenceSession(work, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        assert np.abs(o_ref - o_new).max() < 1e-6, f"{name}: split not FP32-equivalent"
        quant_input = work
    reader = MultiInputCalibReader(net_calib, quant_input)
    quantize_static(
        model_input=quant_input,
        model_output=dst,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt16,
        weight_type=weight_type,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        calibrate_method=calib_method,
    )
    if work and os.path.exists(work):
        os.remove(work)
    # cosine check vs FP32 on one calib sample
    reader.rewind()
    feed = reader.get_next()
    sess_fp = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    of = sess_fp.run(None, feed)[0]
    oq = sess_q.run(None, feed)[0]
    md = float(np.max(np.abs(of.astype(np.float64) - oq.astype(np.float64))))
    # node counts
    qm = onnx.load(dst, load_external_data=False)
    nq = sum(1 for n in qm.graph.node if n.op_type == "QuantizeLinear")
    ndq = sum(1 for n in qm.graph.node if n.op_type == "DequantizeLinear")
    nconv = sum(1 for n in qm.graph.node if n.op_type == "Conv")
    return {"cos": cosine(of, oq), "maxdiff": md, "Q": nq, "DQ": ndq, "Conv": nconv,
            "shape": list(oq.shape)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=MODELS_FP32)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--calib-dir", default=DEFAULT_CALIB)
    ap.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    ap.add_argument("--frames", nargs="+", default=["beauty", "bosphorus", "jockey"])
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 32, 52])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--weight-type", choices=["int16", "int8"], default="int16")
    ap.add_argument("--calibrate-method", choices=["minmax", "entropy", "percentile", "distribution"], default="minmax")
    ap.add_argument("--calib-source", choices=["png", "yuv"], default="png")
    ap.add_argument("--calib-qps", nargs="+", type=int, default=[0, 16, 32, 48, 63])
    ap.add_argument("--calib-frames", nargs="+", type=int, default=[0, 100, 200])
    ap.add_argument("--skip-calib", action="store_true")
    ap.add_argument("--target-qp", type=int, default=None,
                    help="if set, quantize using ONLY this qp's calibration data "
                         "(per-qp PTQ: narrowest activation range, lowest error). "
                         "Output dir is suffixed _qp{qp}.")
    args = ap.parse_args()

    wt = QuantType.QInt16 if args.weight_type == "int16" else QuantType.QInt8
    cm = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
          "percentile": CalibrationMethod.Percentile, "distribution": CalibrationMethod.Distribution}[args.calibrate_method]

    if not args.skip_calib:
        print(f"== dumping calibration -> {args.calib_dir}")
        if args.calib_source == "yuv":
            dump_calibration_yuv(args.models_dir, CALIB_SEQS, args.calib_qps,
                                 args.crop, args.calib_frames, args.calib_dir)
        else:
            dump_calibration(args.models_dir, args.frames_dir, args.frames,
                             args.qps, args.crop, args.calib_dir)

    qp_filter = args.target_qp
    if qp_filter is not None:
        args.out_dir = f"{args.out_dir}_qp{qp_filter}"

    # full copy of model dir (npy tables + untouched inter nets)
    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.models_dir):
        src = os.path.join(args.models_dir, f)
        d = os.path.join(args.out_dir, f)
        if os.path.isfile(src) and not os.path.exists(d):
            shutil.copy2(src, d)

    print(f"== quantizing 10 intra nets: act=QInt16 weight={args.weight_type} fmt=QDQ per-channel"
          f"{' target_qp=' + str(qp_filter) if qp_filter is not None else ''}")
    print(f"{'model':30s} {'cos':>7s} {'maxdiff':>10s} {'Q':>4s} {'DQ':>4s} {'Conv':>5s}  shape")
    for name in INTRA_NETS:
        st = quantize_one(name, args.models_dir, args.calib_dir, args.out_dir, wt, cm, qp_filter)
        print(f"{name:30s} {st['cos']:7.5f} {st['maxdiff']:10.4e} "
              f"{st['Q']:4d} {st['DQ']:4d} {st['Conv']:5d}  {st['shape']}")
    print("done ->", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
