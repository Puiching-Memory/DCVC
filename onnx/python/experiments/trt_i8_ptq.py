#!/usr/bin/env python3
"""ModelOpt INT8 PTQ (QDQ) of the full DCVC-RT intra network set for TRT.

Quantizes the 10 ONNX models used by the C intra end-to-end pipeline
(onnx/src/cpu_intra_pipeline.c + cpu_ar_codec.c):

    intra_analysis_standard, intra_hyper_enc, hyper_dec, y_prior_fusion,
    y_spatial_prior_reduction, y_spatial_prior_adaptor_{1,2,3},
    y_spatial_prior, intra_synthesis

Calibration data:
  - 7 entropy nets: existing dumps in calib_data_rd/<net>/*.npy (input "in0")
  - 3 remaining nets: calib_trt_i8/<net>/sample_*.npz (keys = input names),
    produced by trt_i8_dump_calib.py

high_precision_dtype=fp32 keeps all non-quantized ops in FP32 so the QDQ
models still run on the CPU ORT used by the C closed-loop tests (and stay
directly comparable to the FP32 baseline). For TRT deployment the same QDQ
scales apply; TRT picks INT8 kernels for the Q/DQ regions.

Produces <out-dir>: full copy of --models-dir with the 10 onnx replaced by
their INT8 QDQ versions (npy tables copied untouched).

Usage:
  .venv/bin/python onnx/python/trt_i8_ptq.py [--method entropy]
      [--models-dir ../models_fp32] [--out-dir ../models_trt_i8]
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from modelopt.onnx.quantization import quantize

REPO = os.path.dirname(os.path.abspath(__file__))

INTRA_NETS_RD = [  # single-input nets with existing calib_data_rd npy dumps
    "hyper_dec",
    "y_prior_fusion",
    "y_spatial_prior_reduction",
    "y_spatial_prior_adaptor_1",
    "y_spatial_prior_adaptor_2",
    "y_spatial_prior_adaptor_3",
    "y_spatial_prior",
]
INTRA_NETS_NPZ = [  # dumped by trt_i8_dump_calib.py
    "intra_analysis_standard",
    "intra_hyper_enc",
    "intra_synthesis",
]


class NpyReader(CalibrationDataReader):
    def __init__(self, npy_dir, input_name="in0"):
        self.files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        assert self.files, f"no npy in {npy_dir}"
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


class NpzReader(CalibrationDataReader):
    def __init__(self, npz_dir):
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        assert self.files, f"no npz in {npz_dir}"
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        z = np.load(self.files[self.idx])
        self.idx += 1
        return {k: z[k].astype(np.float32) for k in z.files}

    def rewind(self):
        self.idx = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_fp32")))
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(REPO, "..", "models_trt_i8")))
    ap.add_argument("--calib-rd", default=os.path.join(REPO, "calib_data_rd"))
    ap.add_argument("--calib-npz", default=os.path.join(REPO, "calib_trt_i8"))
    ap.add_argument("--method", default="entropy",
                    choices=["max", "entropy", "percentile", "mse"])
    ap.add_argument("--nets", nargs="+", default=INTRA_NETS_RD + INTRA_NETS_NPZ)
    ap.add_argument("--exclude", default="",
                    help="semicolon-separated net:node1,node2 entries passed to "
                         "modelopt nodes_to_exclude, e.g. "
                         "intra_synthesis:node_conv2d;hyper_dec:node_conv2d_16")
    args = ap.parse_args()

    exclude_map = {}
    for item in args.exclude.split(";"):
        if not item.strip():
            continue
        net, nodes = item.split(":")
        exclude_map[net.strip()] = [n.strip() for n in nodes.split(",") if n.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    # Copy everything first (npy tables, untouched onnx), then replace targets.
    for f in glob.glob(os.path.join(args.models_dir, "*")):
        base = os.path.basename(f)
        dst = os.path.join(args.out_dir, base)
        if os.path.isfile(f) and not os.path.exists(dst):
            shutil.copy2(f, dst)

    for net in args.nets:
        src = os.path.join(args.models_dir, net + ".onnx")
        dst = os.path.join(args.out_dir, net + ".onnx")
        if net in INTRA_NETS_NPZ:
            reader = NpzReader(os.path.join(args.calib_npz, net))
        else:
            reader = NpyReader(os.path.join(args.calib_rd, net))
        print(f"[{net}] calib_samples={len(reader.files)} method={args.method} "
              f"exclude={exclude_map.get(net, [])}")
        quantize(
            onnx_path=src,
            quantize_mode="int8",
            calibration_data_reader=reader,
            calibration_method=args.method,
            high_precision_dtype="fp32",
            output_path=dst,
            nodes_to_exclude=exclude_map.get(net),
            log_level="WARNING",
        )
        print(f"[{net}] -> {dst}")

    print("done")


if __name__ == "__main__":
    main()
