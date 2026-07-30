#!/usr/bin/env python3
"""Build DCVC-RK ONNX subnets -> INT8 (w8a8) RKNN, fully automated.

Same single-source-of-truth shapes.json pipeline as build_rknn.py, but with
do_quantization=True (per-channel weight s8 + per-tensor act u8). Calibration
data is synthesized with representative distributions (the QAT path will make
weights robust to quantization, so exact calibration matters less; for a true
PTQ-only accuracy number you'd dump real-frame activations here instead).

INT8 dataset format (learned the hard way): RKNN expects a .txt FILE listing
calibration samples, one sample per line; single-input = one .npy path per
line, multi-input = N .npy paths space-separated (in input order). .npz is
NOT supported ("Unsupport file").

Usage:
  python DCVC-family/DCVC-RK/build_rknn_i8.py                # all fused subnets
  python DCVC-family/DCVC-RK/build_rknn_i8.py --verify       # + int8-sim vs fp32 cos
  python DCVC-family/DCVC-RK/build_rknn_i8.py --subnets intra_synthesis
"""
import argparse, json, os, sys, tempfile, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))          # .../DCVC-family/DCVC-RK
ROOT = os.path.dirname(os.path.dirname(HERE))              # workspace root (onnx/, checkpoints/, outputs)
import numpy as np
import onnx
import onnxruntime as ort

ONNX_DIR = os.path.join(ROOT, "onnx", "models_rk")
SHAPES_JSON = os.path.join(ONNX_DIR, "shapes.json")
TARGET = os.environ.get("RKNN_TARGET", "rk3588")
MAX_OPSET = 19

NPY_SRC_DIR = os.path.join(ROOT, "onnx", "models")
# Runtime-required constant tables (CDF entropy tables + q_scale + qbanks).
# Loaded by onnx/src/cpu_{intra,inter}_pipeline.c from model_dir root -- they MUST
# sit next to the .rknn files or end-to-end encode/decode fails at create() time.
NPY_FILES = [
    "bitest_cdf.npy", "bitest_cdf_length.npy", "bitest_offset.npy",
    "gaussian_cdf.npy", "gaussian_cdf_length.npy", "gaussian_offset.npy",
    "q_scale_enc.npy", "q_scale_dec.npy",
    "q_encoder.npy", "q_decoder.npy", "q_feature.npy", "q_recon.npy",
]


def stage_npy(out_dir):
    """Copy the 12 runtime constant tables into the deploy dir (.rknn siblings)."""
    import shutil
    n = 0
    for f in NPY_FILES:
        s = os.path.join(NPY_SRC_DIR, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out_dir, f))
            n += 1
    return n


# ----------------------------------------------------------- calibration ----
# representative scale per input ROLE (keeps act ranges realistic enough for the
# quantizer; not real-data accurate -> QAT training removes this dependence).
def _scale_for(shape):
    c = shape[1] if len(shape) >= 2 else 1
    # image-like (3ch) -> normalized ~N(0,1); latent/feature (>=64ch) ~ N(0,~8);
    # quant_step (1x1) ~ small
    if len(shape) == 4 and shape[2] == 1:        # quant_step (1,C,1,1)
        return 0.5
    if c == 3:                                   # RGB image (normalized)
        return 1.0
    return 8.0                                   # latent / context feature


def gen_calib(shapes, n=8, seed=0):
    """Write a dataset.txt + npy files; return (txt_path, calib_dir)."""
    calib = tempfile.mkdtemp(prefix="_calib_i8_")
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n):
        ps = []
        for j, s in enumerate(shapes):
            p = os.path.join(calib, f"s{i}_{j}.npy")
            np.save(p, rng.standard_normal(s).astype(np.float32) * _scale_for(s))
            ps.append(p)
        lines.append(" ".join(ps))
    txt = os.path.join(calib, "dataset.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    return txt


# ------------------------------------------------------------------- bake ----
def bake_onnx(name, shapes):
    m = onnx.load(os.path.join(ONNX_DIR, name + ".onnx"))
    for inp, shp in zip(m.graph.input, shapes):
        dims = inp.type.tensor_type.shape.dim
        for dim, val in zip(dims, shp):
            dim.ClearField("dim_param")
            dim.dim_value = int(val)
    for o in m.opset_import:
        if (o.domain or "ai.onnx") == "ai.onnx" and o.version > MAX_OPSET:
            o.version = MAX_OPSET
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix=f"_i8_{name}_")
    os.close(fd)
    onnx.save(m, path)
    return path


def _dup2_capture(logf):
    so, se = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
    return so, se, fd


# ----------------------------------------------------------- i8 build ---
def build_i8(name, shapes, rknn_out):
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_i8log_{name}_")[1]
    baked = bake_onnx(name, shapes)
    txt = gen_calib(shapes)
    so, se, fd = _dup2_capture(logf)
    from rknn.api import RKNN
    r = RKNN(verbose=True)
    rc = None
    try:
        r.config(mean_values=[], std_values=[], target_platform=TARGET,
                 quantized_dtype="w8a8", quantized_method="channel")
        rc = r.load_onnx(model=baked)
        if rc == 0:
            rc = r.build(do_quantization=True, dataset=txt)
            if rc == 0:
                r.export_rknn(rknn_out)
    finally:
        os.dup2(so, 1); os.dup2(se, 2)
        os.close(fd); os.close(so); os.close(se)
        r.release()
    return rc, logf


# ----------------------------------------------------------- verify -------
def _n2h(a):
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def _cos(a, b):
    a = np.asarray(a, np.float32).ravel(); b = np.asarray(b, np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify_i8(name, shapes):
    """Cosine of int8 RKNN host-sim vs fp32 ORT on the SAME random input."""
    baked = bake_onnx(name, shapes)
    m = onnx.load(baked)
    rng = np.random.default_rng(0)
    feed = {i.name: rng.standard_normal(s).astype(np.float32) * _scale_for(s)
            for i, s in zip(m.graph.input, shapes)}
    so_ = ort.SessionOptions(); so_.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), sess_options=so_,
                                providers=["CPUExecutionProvider"])
    ort_out = [np.asarray(o, np.float32) for o in sess.run(None, feed)]
    txt = gen_calib(shapes)
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_i8ver_{name}_")[1]
    from rknn.api import RKNN
    so_fd, se_fd, fd = _dup2_capture(logf)
    r = RKNN(verbose=False); cos = -1.0
    try:
        r.config(mean_values=[], std_values=[], target_platform=TARGET,
                 quantized_dtype="w8a8", quantized_method="channel")
        r.load_onnx(model=baked)
        r.build(do_quantization=True, dataset=txt)
        r.init_runtime(target=None)
        rk = r.inference(inputs=[_n2h(v) for v in feed.values()])
        cos = min(_cos(a, b) for a, b in zip(ort_out, rk))
    finally:
        os.dup2(so_fd, 1); os.dup2(se_fd, 2)
        r.release()
    return cos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "dcvc_rk_1080p_rk3588_i8"))
    ap.add_argument("--subnets", nargs="*", default=None)
    ap.add_argument("--verify", action="store_true", help="int8-sim vs fp32 cosine")
    ap.add_argument("--pack", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = json.load(open(SHAPES_JSON))
    shapes_map = manifest["subnets"]
    names = args.subnets or list(shapes_map)
    print(f"== INT8 build {len(names)} subnets (w8a8/channel, target={TARGET}) ==")
    hdr = f"{'subnet':28s} {'rc':>2s}  cos"
    print(hdr); print("-" * len(hdr))
    for n in names:
        shapes = shapes_map[n]
        rknn_path = os.path.join(args.out_dir, n + ".rknn")
        rc, _ = build_i8(n, shapes, rknn_path)
        cos = verify_i8(n, shapes) if args.verify else None
        cstr = f"{cos:.4f}" if cos is not None else "  -"
        print(f"{n:28s} {rc:2d}  {cstr}")
    if args.pack:
        stage_npy(args.out_dir)
        with zipfile.ZipFile(args.pack, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(args.out_dir)):
                p = os.path.join(args.out_dir, fn)
                if os.path.isfile(p):
                    z.write(p, os.path.relpath(p, ROOT))
            for src in (os.path.join(HERE, "export_rk.py"),
                        os.path.join(HERE, "build_rknn_i8.py"),
                        os.path.join(HERE, "qat.py")):
                if os.path.exists(src):
                    z.write(src, os.path.relpath(src, ROOT))
        print(f"packed -> {args.pack}")


if __name__ == "__main__":
    main()
