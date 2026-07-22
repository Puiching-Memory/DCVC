#!/usr/bin/env python3
"""Verify the INT8 PTQ models: cross-config bit-exactness and scale-index agreement.

Part a (bit-exact): for each quantized model, run the same inputs under the 4
combinations of intra_op_num_threads in {1, 8} x graph_optimization_level in
{ORT_DISABLE_ALL, ORT_ENABLE_ALL} and require the outputs to be bit-identical
(raw byte compare). Integer conv accumulation is order-independent, so this
must hold for the INT8 models; the FP32 originals are run as a reference only
(FP32 accumulation order is scheduling-dependent, so they may FAIL - that is
exactly the desync mechanism this PTQ prototype eliminates).

Part b (RD proxy): for the nets whose outputs contain the rANS scales
(y_prior_fusion: channels 2..257 of params_fusion; y_spatial_prior: channels
0..255 of its output), compute the C-side scale index
    idx = clamp(floor((log(clamp(s,0.11,16)) - log(0.11)) * 127/(log(16)-log(0.11))), 0, 127)
(cpu_ar_codec.c build_index_enc/dec) for both the INT8 and the FP32 model on
identical inputs, and report the fraction of positions where the indexes
agree. rANS stays in sync as long as both ends run INT8 (part a); this metric
quantifies the RD impact of quantization vs the FP32 reference.

Usage:
  python ptq_verify.py [--models-dir ../models] [--int8-dir ../models_int8]
                       [--calib-dir calib_data] [--n-bitexact 3] [--n-index 0]
"""
import argparse
import glob
import os

import numpy as np
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

# model -> (scale channel slice of the output, description)
SCALE_OUTPUTS = {
    "y_prior_fusion": (slice(2, 2 + 256), "params_fusion scales (round 0)"),
    "y_spatial_prior": (slice(0, 256), "spatial_prior scales (rounds 1-3)"),
}

LOG_SCALE_MIN = np.float32(np.log(np.float32(0.11)))
LOG_STEP_RECIP = np.float32(1.0) / ((np.float32(np.log(np.float32(16.0))) - LOG_SCALE_MIN) / np.float32(127.0))


def scale_index(scales):
    """C build_index_enc/dec, float32."""
    s = np.asarray(scales, dtype=np.float32)
    s = np.clip(s, np.float32(0.11), np.float32(16.0))
    v = (np.log(s) - LOG_SCALE_MIN) * LOG_STEP_RECIP
    idx = np.floor(v).astype(np.int32)
    return np.clip(idx, 0, 127)


def make_session(path, threads, opt_level):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = opt_level
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def bitexact_one(model_path, inputs):
    """Run inputs under the 4 configs; return (all_identical, per-config results)."""
    configs = [
        (1, ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        (1, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        (8, ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        (8, ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ]
    outs = {}
    for threads, opt in configs:
        sess = make_session(model_path, threads, opt)
        key = f"t{threads}/{'all' if opt == ort.GraphOptimizationLevel.ORT_ENABLE_ALL else 'off'}"
        outs[key] = [sess.run(None, {"in0": x})[0].tobytes() for x in inputs]
    ref = None
    ok = True
    for key, res in outs.items():
        if ref is None:
            ref = res
        elif res != ref:
            ok = False
    return ok, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.normpath(os.path.join(REPO, "..", "models")))
    ap.add_argument("--int8-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_int8")))
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_data"))
    ap.add_argument("--n-bitexact", type=int, default=3,
                    help="calib inputs per model for the bit-exact test")
    ap.add_argument("--n-index", type=int, default=0,
                    help="calib inputs per scale model for index agreement; 0 = all")
    args = ap.parse_args()

    # ---- Part a: 4-config bit-exactness ----
    print("=== Part a: bit-exactness across threads x graph-opt (byte-identical outputs) ===")
    print(f"{'model':36s} {'INT8':6s} {'FP32(ref)':10s}")
    for name in TARGET_NETS:
        files = sorted(glob.glob(os.path.join(args.calib_dir, name, "*.npy")))[: args.n_bitexact]
        inputs = [np.load(f).astype(np.float32) for f in files]
        ok8, _ = bitexact_one(os.path.join(args.int8_dir, name + ".onnx"), inputs)
        ok32, _ = bitexact_one(os.path.join(args.models_dir, name + ".onnx"), inputs)
        print(f"{name:36s} {'PASS' if ok8 else 'FAIL':6s} "
              f"{'identical' if ok32 else 'DIFFERS':10s}")

    # ---- Part b: scale-index agreement INT8 vs FP32 ----
    # NOTE: the plain agreement ratio is dominated by the clamp floor in
    # build_index (86% of FP32 scales are <= 0.11 and map to index 0; a small
    # scale error crossing 0.11 flips index 0 <-> 1/2 whose CDFs are nearly
    # identical). mean|didx| and the scale abs error give the complementary
    # magnitude view; the end-to-end RD table is the authoritative metric.
    print("\n=== Part b: rANS scale-index agreement INT8 vs FP32 ===")
    print(f"{'model':36s} {'agree%':>9s} {'mismatch/total':>18s} {'mean|didx|':>11s} {'scaleErr':>9s}")
    total_mm = total_n = 0
    for name, (ch_slice, desc) in SCALE_OUTPUTS.items():
        files = sorted(glob.glob(os.path.join(args.calib_dir, name, "*.npy")))
        if args.n_index:
            files = files[: args.n_index]
        sess8 = make_session(os.path.join(args.int8_dir, name + ".onnx"), 0,
                             ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        sess32 = make_session(os.path.join(args.models_dir, name + ".onnx"), 0,
                              ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        mm = n = 0
        didx_sum = 0.0
        serr_sum = 0.0
        for f in files:
            x = np.load(f).astype(np.float32)
            o8 = sess8.run(None, {"in0": x})[0]
            o32 = sess32.run(None, {"in0": x})[0]
            s8, s32 = o8[:, ch_slice], o32[:, ch_slice]
            i8 = scale_index(s8)
            i32 = scale_index(s32)
            mm += int(np.count_nonzero(i8 != i32))
            didx_sum += float(np.abs(i8 - i32).sum())
            serr_sum += float(np.abs(s8.astype(np.float64) - s32.astype(np.float64)).sum())
            n += i8.size
        total_mm += mm
        total_n += n
        print(f"{name:36s} {100.0 * (n - mm) / n:8.4f}% {mm:9d}/{n:<9d} "
              f"{didx_sum / n:11.4f} {serr_sum / n:9.5f}  ({desc})")
    print(f"{'OVERALL':36s} {100.0 * (total_n - total_mm) / total_n:8.4f}% "
          f"{total_mm:9d}/{total_n:<9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
