#!/usr/bin/env python3
"""Dump ModelOpt PTQ calibration inputs for the 3 intra nets not covered by
calib_data_rd (intra_analysis_standard, intra_hyper_enc, intra_synthesis).

Reuses the FP32 encoder dataflow from ptq_dump_calib.py (bit-verified against
the C pipeline) and the same (frame, qp) sample list from calib_data_rd's
manifest, so the calibration distribution matches the existing entropy-net
calibration data.

Output layout (npz, keys = ONNX input names):
  <out>/<net>/sample_<i>.npz

Usage:
  .venv/bin/python onnx/python/trt_i8_dump_calib.py [--out calib_trt_i8]
"""
import argparse
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, REPO)

from ptq_dump_calib import Encoder, rgb_to_ycbcr_c  # noqa: E402

DEFAULT_MODELS = os.path.normpath(os.path.join(REPO, "..", "models_fp32"))
CALIB_RD = os.path.join(REPO, "calib_data_rd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--calib-rd", default=CALIB_RD,
                    help="existing entropy-net calib dir (manifest + x npys)")
    ap.add_argument("--out", default=os.path.join(REPO, "calib_trt_i8"))
    args = ap.parse_args()

    enc = Encoder(args.models_dir)
    q_scale_dec = np.load(os.path.join(args.models_dir, "q_scale_dec.npy"))

    manifest = json.load(open(os.path.join(args.calib_rd, "manifest.json")))

    for net in ["intra_analysis_standard", "intra_hyper_enc", "intra_synthesis"]:
        os.makedirs(os.path.join(args.out, net), exist_ok=True)

    for i, entry in enumerate(manifest):
        frame, qp, crop = entry["frame"], entry["qp"], entry["crop"]
        x_path = os.path.join(args.calib_rd, "x", f"{frame}_crop{crop}.npy")
        x = np.load(x_path)  # [1,3,H,W] float32 RGB

        x_ycbcr = rgb_to_ycbcr_c(x[0])[None]
        qenc = enc.q_scale_enc[qp:qp + 1].astype(np.float32)
        qdec = q_scale_dec[qp:qp + 1].astype(np.float32)

        res = enc.encode_sample(x, qp, dump=lambda *a: None)

        np.savez(os.path.join(args.out, "intra_analysis_standard", f"sample_{i}.npz"),
                 in0=np.ascontiguousarray(x_ycbcr), in1=np.ascontiguousarray(qenc))
        np.savez(os.path.join(args.out, "intra_hyper_enc", f"sample_{i}.npz"),
                 in0=np.ascontiguousarray(res["y"]))
        np.savez(os.path.join(args.out, "intra_synthesis", f"sample_{i}.npz"),
                 in0=np.ascontiguousarray(res["y_hat"]), in1=np.ascontiguousarray(qdec))
        print(f"[sample_{i}] frame={frame} qp={qp} y{list(res['y'].shape)} "
              f"y_hat{list(res['y_hat'].shape)}")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
