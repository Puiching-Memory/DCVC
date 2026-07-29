#!/usr/bin/env python3
"""Hybrid pack: native ONNX INT8 backbone + FXP entropy nets.

Measures the split we agreed on:
  - Backbone (analysis / hyper_enc / synthesis): ORT QLinearConv (MLAS int8)
  - Entropy nets: keep com.dcvc FXP from the default models/ pack

Steps:
  1) Dump calib inputs from FP32 backbone on real frames
  2) quantize_static (QOperator, u8 act / s8 per-channel weights)
  3) Assemble models_hybrid_i8bb/ = copy(models/) + replace 3 backbone nets
  4) RD + wall-time vs baseline FXP pack (FP32 backbone + FXP entropy)

Usage:
  uv run python python/ptq_hybrid_backbone_i8.py \\
      --frames-dir rd_frames --frames beauty bosphorus jockey \\
      --qps 12 22 32 42 52 --crop 256
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, ".."))
sys.path.insert(0, REPO)
from ptq_dump_calib import Encoder, load_frame, rgb_to_ycbcr_c  # noqa: E402

BACKBONE_NETS = [
    "intra_analysis_standard",
    "intra_hyper_enc",
    "intra_synthesis",
]
SOURCE_IR_VERSION = 8
C_BIN = os.path.join(ONNX_DIR, "build", "test_cpu_end2end")


class NpzCalibrationReader(CalibrationDataReader):
    """Feeds .npz dicts (keys = ONNX input names)."""

    def __init__(self, npz_dir: str):
        self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no npz in {npz_dir}")
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.files):
            return None
        z = np.load(self.files[self.idx])
        self.idx += 1
        return {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in z.files}

    def rewind(self):
        self.idx = 0


def dump_backbone_calib(fp32_dir, frames_dir, frames, qps, crop, out_dir):
    """Dump analysis/hyper_enc/synthesis call inputs as .npz."""
    enc = Encoder(fp32_dir)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    synth = ort.InferenceSession(
        os.path.join(fp32_dir, "intra_synthesis.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    q_dec = np.load(os.path.join(fp32_dir, "q_scale_dec.npy"))

    for net in BACKBONE_NETS:
        os.makedirs(os.path.join(out_dir, net), exist_ok=True)

    if isinstance(crop, int):
        cw = ch = crop
    else:
        cw, ch = crop

    manifest = []
    sid = 0
    for frame in frames:
        x = load_frame(frames_dir, frame, (cw, ch))
        for qp in qps:
            x_ycbcr = rgb_to_ycbcr_c(x[0])[None].astype(np.float32)
            qenc = enc.q_scale_enc[qp : qp + 1].astype(np.float32)
            qdec = q_dec[qp : qp + 1].astype(np.float32)

            np.savez(
                os.path.join(out_dir, "intra_analysis_standard", f"s{sid:02d}.npz"),
                in0=x_ycbcr,
                in1=qenc,
            )

            y = enc.sess["intra_analysis_standard"].run(
                None, {"in0": x_ycbcr, "in1": qenc}
            )[0].astype(np.float32)
            y = np.clip(y, -128.0, 127.0).astype(np.float32)

            np.savez(
                os.path.join(out_dir, "intra_hyper_enc", f"s{sid:02d}.npz"),
                in0=y,
            )

            # Full AR path so synthesis calib sees real y_hat, not raw y.
            def _noop(*_a, **_k):
                return None

            res = enc.encode_sample(x, qp, _noop)
            y_hat = res["y_hat"].astype(np.float32)
            np.savez(
                os.path.join(out_dir, "intra_synthesis", f"s{sid:02d}.npz"),
                in0=y_hat,
                in1=qdec,
            )
            # Smoke: synthesis must accept the feed.
            _ = synth.run(None, {"in0": y_hat, "in1": qdec})

            manifest.append({"sample": f"s{sid:02d}", "frame": frame, "qp": qp})
            print(f"[s{sid:02d}] {frame} qp={qp} y={list(y.shape)} y_hat={list(y_hat.shape)}")
            sid += 1

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"dumped {sid} samples -> {out_dir}")
    return sid


def check_quantized(path):
    model = onnx.load(path, load_external_data=False)
    producer = {o: n for n in model.graph.node for o in n.output}
    n_q = n_dq = n_conv = n_qconv = n_fp32 = 0
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
                n_fp32 += 1
    return {
        "Q": n_q,
        "DQ": n_dq,
        "QLConv": n_qconv,
        "Conv": n_conv,
        "fp32Conv": n_fp32,
        "ir": model.ir_version,
    }


def quantize_backbone(fp32_dir, calib_dir, work_dir, calibrate_method, percentile):
    os.makedirs(work_dir, exist_ok=True)
    method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[calibrate_method]
    extra = {}
    if method == CalibrationMethod.Percentile:
        extra["CalibPercentile"] = percentile

    print(f"{'model':28s} {'Q':>4} {'DQ':>4} {'QLConv':>7} {'fp32':>5}")
    for name in BACKBONE_NETS:
        src = os.path.join(fp32_dir, name + ".onnx")
        dst = os.path.join(work_dir, name + ".onnx")
        reader = NpzCalibrationReader(os.path.join(calib_dir, name))
        quantize_static(
            model_input=src,
            model_output=dst,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            calibrate_method=method,
            extra_options=extra or None,
        )
        model = onnx.load(dst, load_external_data=False)
        if model.ir_version != SOURCE_IR_VERSION:
            model.ir_version = SOURCE_IR_VERSION
            onnx.save(model, dst)
        st = check_quantized(dst)
        print(
            f"{name:28s} {st['Q']:4d} {st['DQ']:4d} {st['QLConv']:7d} {st['fp32Conv']:5d}"
        )
        if st["fp32Conv"]:
            raise RuntimeError(f"{name}: residual FP32 Conv after QOperator PTQ")
        # sanity run
        reader.rewind()
        feed = reader.get_next()
        out = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, feed)[0]
        print(f"  sanity out {list(out.shape)} dtype={out.dtype}")


def assemble_hybrid(baseline_dir, i8_backbone_dir, out_dir):
    if os.path.abspath(baseline_dir) == os.path.abspath(out_dir):
        raise ValueError("out_dir must differ from baseline")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(baseline_dir, out_dir)
    for name in BACKBONE_NETS:
        src = os.path.join(i8_backbone_dir, name + ".onnx")
        dst = os.path.join(out_dir, name + ".onnx")
        shutil.copy2(src, dst)
        print(f"hybrid: replaced {name}")
    # Verify entropy still FXP, backbone native QLinearConv
    for name in ["hyper_dec", "y_prior_fusion"]:
        m = onnx.load(os.path.join(out_dir, name + ".onnx"))
        domains = sorted({n.domain or "" for n in m.graph.node})
        print(f"  {name} domains={domains}")
    for name in BACKBONE_NETS:
        st = check_quantized(os.path.join(out_dir, name + ".onnx"))
        print(f"  {name} QLConv={st['QLConv']} fp32Conv={st['fp32Conv']}")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def run_one(model_dir, x_path, rec_path, h, w, qp):
    t0 = time.perf_counter()
    proc = subprocess.run(
        [C_BIN, model_dir, str(h), str(w), str(qp), x_path, rec_path],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0 or "PASS" not in proc.stdout:
        raise RuntimeError(
            f"roundtrip failed\nCMD: {C_BIN} {model_dir} ...\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    m = re.search(r"stream_size=(\d+) bytes", proc.stdout)
    md = re.search(r"x_hat max_diff=([0-9.eE+-]+)", proc.stdout)
    return int(m.group(1)), float(md.group(1)), wall


def eval_rd(baseline_dir, hybrid_dir, frames_dir, frames, qps, crop, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cw = ch = crop if isinstance(crop, int) else crop[0]
    rows = []
    print(
        f"\n{'frame':12s} {'qp':>4} {'Bbase':>8} {'Bhyb':>8} {'ratio':>7} "
        f"{'dPSNR':>8} {'t_base':>8} {'t_hyb':>8} {'speedup':>7}"
    )
    for frame in frames:
        x = load_frame(frames_dir, frame, (cw, ch))
        x_path = os.path.join(out_dir, f"{frame}_{cw}x{ch}.npy")
        np.save(x_path, x)
        for qp in qps:
            rec_b = os.path.join(out_dir, f"{frame}_qp{qp}_base_rec.npy")
            rec_h = os.path.join(out_dir, f"{frame}_qp{qp}_hyb_rec.npy")
            b0, d0, t0 = run_one(baseline_dir, x_path, rec_b, ch, cw, qp)
            b1, d1, t1 = run_one(hybrid_dir, x_path, rec_h, ch, cw, qp)
            p0 = psnr(x, np.load(rec_b))
            p1 = psnr(x, np.load(rec_h))
            row = {
                "frame": frame,
                "qp": qp,
                "bytes_base": b0,
                "bytes_hybrid": b1,
                "byte_ratio": b1 / max(b0, 1),
                "psnr_base": p0,
                "psnr_hybrid": p1,
                "dpsnr": p1 - p0,
                "encdec_maxdiff_base": d0,
                "encdec_maxdiff_hybrid": d1,
                "wall_s_base": t0,
                "wall_s_hybrid": t1,
                "speedup": t0 / t1 if t1 > 0 else 0.0,
            }
            rows.append(row)
            print(
                f"{frame:12s} {qp:4d} {b0:8d} {b1:8d} {row['byte_ratio']:7.4f} "
                f"{row['dpsnr']:+8.4f} {t0:8.2f} {t1:8.2f} {row['speedup']:7.3f}"
            )
    with open(os.path.join(out_dir, "rd_results.json"), "w") as f:
        json.dump(rows, f, indent=1)

    br = float(np.mean([r["byte_ratio"] for r in rows]))
    dp = float(np.mean([r["dpsnr"] for r in rows]))
    sp = float(np.mean([r["speedup"] for r in rows]))
    print("\n=== SUMMARY: hybrid (native i8 backbone + FXP entropy) vs baseline ===")
    print(f"  mean byte_ratio = {br:.4f}  ({(br - 1) * 100:+.2f}% bitrate)")
    print(f"  mean ΔPSNR      = {dp:+.4f} dB")
    print(f"  mean speedup    = {sp:.3f}×  (wall, end-to-end encode+decode)")
    print(f"  max enc↔dec Δ   hybrid={max(r['encdec_maxdiff_hybrid'] for r in rows):.1e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=os.path.join(ONNX_DIR, "models"),
                    help="FXP entropy + FP32 backbone (current default pack)")
    ap.add_argument("--fp32-dir", default=os.path.join(ONNX_DIR, "models_fp32"),
                    help="all-FP32 pack used to dump calib / quantize backbone")
    ap.add_argument("--calib-dir", default=os.path.join(REPO, "calib_backbone_i8"))
    ap.add_argument("--i8-work-dir", default=os.path.join(ONNX_DIR, "models_backbone_i8"))
    ap.add_argument("--hybrid-dir", default=os.path.join(ONNX_DIR, "models_hybrid_i8bb"))
    ap.add_argument("--out", default=os.path.join(REPO, "rd_hybrid_i8bb"))
    ap.add_argument("--frames-dir", default=os.path.join(ONNX_DIR, "rd_frames"))
    ap.add_argument("--frames", nargs="+", default=["beauty", "bosphorus", "jockey"])
    ap.add_argument("--qps", nargs="+", type=int, default=[12, 22, 32, 42, 52])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--calibrate-method", choices=["minmax", "entropy", "percentile"],
                    default="percentile")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--skip-dump", action="store_true")
    ap.add_argument("--skip-quant", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(C_BIN):
        print(f"missing {C_BIN}; build onnx/ first", file=sys.stderr)
        return 1

    if not args.skip_dump:
        print("=== 1) dump backbone calib ===")
        dump_backbone_calib(
            args.fp32_dir, args.frames_dir, args.frames, args.qps, args.crop, args.calib_dir
        )

    if not args.skip_quant:
        print("=== 2) native ORT INT8 PTQ (QOperator) ===")
        quantize_backbone(
            args.fp32_dir,
            args.calib_dir,
            args.i8_work_dir,
            args.calibrate_method,
            args.percentile,
        )
        print("=== 3) assemble hybrid pack ===")
        assemble_hybrid(args.baseline_dir, args.i8_work_dir, args.hybrid_dir)

    if not args.skip_eval:
        print("=== 4) RD + wall-time ===")
        eval_rd(
            args.baseline_dir,
            args.hybrid_dir,
            args.frames_dir,
            args.frames,
            args.qps,
            args.crop,
            args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
