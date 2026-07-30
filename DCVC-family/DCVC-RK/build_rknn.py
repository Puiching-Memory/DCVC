#!/usr/bin/env python3
"""Build DCVC-RK ONNX subnets -> RKNN, fully automated and self-checking.

Single source of truth for input shapes: `shapes.json` (written by export_rk.py from
the model definitions). This script NEVER hardcodes shapes -- it reads them from the
manifest, so changing a channel width in the model auto-propagates to the baked RKNN
graph. This eliminates the class of bug where a stale hardcoded shape (e.g. an old
384 quant-step width) is baked into a model whose actual tensor is now 256.

Pipeline (single command):
  1. (optional) re-export ONNX + shapes.json from the current model defs
  2. bake each ONNX with concrete dims  ->  load  ->  build  ->  export .rknn
  3. analyze the RKNN verbose log: NPU/CPU op counts, 3-core-fallback layers
     classified by role (depthwise / GLU-expand / 1x1-adaptor / head), memory
  4. (optional) numerics: RKNN host-sim vs ONNXRuntime cosine per subnet
  5. (optional) pack everything into a zip

Usage:
  python DCVC-family/DCVC-RK/build_rknn.py                    # export + build + analyze
  python DCVC-family/DCVC-RK/build_rknn.py --verify           # + numerics check
  python DCVC-family/DCVC-RK/build_rknn.py --pack out.zip     # + zip
  python DCVC-family/DCVC-RK/build_rknn.py --no-reexport      # skip ONNX export, reuse onnx/models_rk
  python DCVC-family/DCVC-RK/build_rknn.py --subnets recon_generation inter_decoder
"""
import argparse
import json
import os
import re
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))          # .../DCVC-family/DCVC-RK
ROOT = os.path.dirname(os.path.dirname(HERE))              # workspace root (onnx/, checkpoints/, outputs)

import numpy as np
import onnx
import onnxruntime as ort

ONNX_DIR = os.path.join(ROOT, "onnx", "models_rk")
SHAPES_JSON = os.path.join(ONNX_DIR, "shapes.json")
TARGET = os.environ.get("RKNN_TARGET", "rk3588")

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


# ---------------------------------------------------------------- reexport ----
def reexport():
    """Run export_rk.py --all in-process so ONNX + shapes.json reflect current model."""
    import importlib
    import export_rk as exp
    importlib.reload(exp)
    sys.argv = ["export_rk.py", "--all", "--out-dir", ONNX_DIR]
    exp.main()


# ------------------------------------------------------------------- bake ----
def bake_onnx(name, shapes, max_opset=19):
    """Assign concrete dims to an ONNX input graph, save to a fresh temp path.
    Returns (temp_path, onnx_model). Concrete shapes come from shapes.json."""
    src = os.path.join(ONNX_DIR, name + ".onnx")
    m = onnx.load(src)
    for inp, shp in zip(m.graph.input, shapes):
        dims = inp.type.tensor_type.shape.dim
        assert len(dims) == len(shp), f"{name}/{inp.name}: rank {len(dims)} != {len(shp)}"
        for dim, val in zip(dims, shp):
            dim.ClearField("dim_param")
            dim.dim_value = int(val)
    for o in m.opset_import:
        if (o.domain or "ai.onnx") == "ai.onnx" and o.version > max_opset:
            o.version = max_opset
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix=f"_rk_{name}_")
    os.close(fd)
    onnx.save(m, path)
    return path


# --------------------------------------------------------------- rknn build ---
def build_one(name, shapes, rknn_out=None):
    """Build a single subnet to .rknn (and export if rknn_out given), capturing the
    verbose log via fd dup2 (RKNN writes its verbose stream to the raw fd, bypassing
    Python's stdout). Builds exactly once."""
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_rklog_{name}_")[1]
    baked = bake_onnx(name, shapes)
    so, se = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    from rknn.api import RKNN
    r = RKNN(verbose=True)
    rc = None
    try:
        os.dup2(fd, 1); os.dup2(fd, 2)
        r.config(mean_values=[], std_values=[], target_platform=TARGET, float_dtype="float16")
        rc = r.load_onnx(model=baked)
        if rc == 0:
            rc = r.build(do_quantization=False)
            if rc == 0 and rknn_out:
                r.export_rknn(rknn_out)
    finally:
        os.dup2(so, 1); os.dup2(se, 2)
        os.close(fd); os.close(so); os.close(se)
        r.release()
    return rc, logf


# ----------------------------------------------------------- log analysis ----
def _layer_role(path):
    if "dc.2" in path or "dc/dc.2" in path:
        return "depthwise3x3"
    if "ffn/ea" in path or "ffn/eb" in path or "ffn.e" in path:
        return "glu_expand"
    if "adaptor" in path:
        return "1x1_adaptor"
    if "head" in path:
        return "head"
    if "down" in path:
        return "downsample"
    return path.split("/")[-1] or "other"


def analyze(log_path):
    """Parse a RKNN verbose build log: fallback layers, op core placement, memory."""
    raw = open(log_path, errors="ignore").read()
    t = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    fb_layers = re.findall(r"Failed to config layer: '(.*?)' using 3Core", t)
    fb_roles = {}
    for p in fb_layers:
        role = _layer_role(p)
        fb_roles[role] = fb_roles.get(role, 0) + 1
    cpu_ops, npu_ops = set(), set()
    for line in t.splitlines():
        m = re.match(r"^.*?\]\s*\d+\s+(\w+)\s+FLOAT16\s+(?:NC1HWC2\s+)?\(?[\w,]*\)?\s*(NPU|CPU)\b", line)
        if m:
            (npu_ops if m.group(2) == "NPU" else cpu_ops).add(m.group(1))
    mem = re.search(r"Total Internal Memory Size:\s*([\d.]+)\s*(KB|MB)", t)
    mem_mb = float(mem.group(1)) * (1 if mem.group(2) == "MB" else 1 / 1024) if mem else 0.0
    compute_cpu = sorted(o for o in cpu_ops if o not in ("InputOperator", "OutputOperator"))
    return len(fb_layers), fb_roles, mem_mb, compute_cpu


# ------------------------------------------------------------- numerics ------
def _n2h(a):
    return np.transpose(a, (0, 2, 3, 1)) if a.ndim == 4 else a


def _cos(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify_one(name, shapes):
    """Cosine similarity of RKNN host-sim vs ONNXRuntime on random input."""
    baked = bake_onnx(name, shapes)
    m = onnx.load(baked)
    rng = np.random.default_rng(0)
    feed = {i.name: rng.standard_normal(s).astype(np.float32) for i, s in zip(m.graph.input, shapes)}
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])
    ort_out = [np.asarray(o, np.float32) for o in sess.run(None, feed)]
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_rkver_{name}_")[1]
    from rknn.api import RKNN
    so_fd, se_fd = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    r = RKNN(verbose=False)
    cos = -1.0
    try:
        os.dup2(fd, 1); os.dup2(fd, 2)
        r.config(mean_values=[], std_values=[], target_platform=TARGET, float_dtype="float16")
        r.load_onnx(model=baked)
        r.build(do_quantization=False)
        r.init_runtime(target=None)
        rk = r.inference(inputs=[_n2h(v) for v in feed.values()])
        cos = min(_cos(a, b) for a, b in zip(ort_out, rk))
    finally:
        os.dup2(so_fd, 1); os.dup2(se_fd, 2)
        r.release()
    return cos


# ------------------------------------------------------------------- pack -----
def pack(out_zip, out_dir, extra_files=()):
    """Zip the whole deploy dir (.rknn + staged .npy constant tables) + sources."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out_dir)):
            p = os.path.join(out_dir, fn)
            if os.path.isfile(p):
                z.write(p, os.path.relpath(p, ROOT))
        for src in extra_files:
            if os.path.exists(src):
                z.write(src, os.path.relpath(src, ROOT))
    return out_zip


# ------------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "dcvc_rk_1080p_rk3588"))
    ap.add_argument("--subnets", nargs="*", default=None)
    ap.add_argument("--no-reexport", action="store_true", help="skip ONNX export; reuse onnx/models_rk")
    ap.add_argument("--verify", action="store_true", help="run RKNN-vs-ORT cosine check")
    ap.add_argument("--pack", default=None, help="pack results into this zip")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_reexport:
        print("== stage 1: re-export ONNX + shapes.json from model defs ==")
        reexport()

    if not os.path.exists(SHAPES_JSON):
        sys.exit(f"ERROR: {SHAPES_JSON} missing. Run without --no-reexport first.")
    manifest = json.load(open(SHAPES_JSON))
    shapes_map = manifest["subnets"]
    names = args.subnets or list(shapes_map)
    print(f"\n== stage 2: build {len(names)} RKNN (target={TARGET}) ==")

    hdr = f"{'subnet':28s} {'rc':>2s} {'fb':>3s} {'mem':>7s} {'CPUops':>7s}  fallback-roles"
    print(hdr); print("-" * len(hdr))
    results = {}
    for n in names:
        shapes = shapes_map[n]
        rknn_path = os.path.join(args.out_dir, n + ".rknn")
        rc, logf = build_one(n, shapes, rknn_out=rknn_path)
        nfb, roles, mem, cpu = analyze(logf)
        role_str = ",".join(f"{k}×{v}" for k, v in sorted(roles.items())) or "-"
        results[n] = dict(rc=rc, fb=nfb, mem=mem, cpu=cpu, roles=roles)
        print(f"{n:28s} {rc:2d} {nfb:3d} {mem:6.0f}MB {len(cpu):7d}  {role_str}"
              + (f"  CPU={','.join(cpu)}" if cpu else ""))

    if args.verify:
        print("\n== stage 3: numerics (RKNN host-sim vs ORT) ==")
        for n in names:
            if results[n]["rc"] != 0:
                continue
            c = verify_one(n, shapes_map[n])
            print(f"  {n:28s} cos={c:.4f}")

    tot_fb = sum(r["fb"] for r in results.values())
    ok = sum(1 for r in results.values() if r["rc"] == 0)
    print(f"\n{ok}/{len(names)} built, {tot_fb} total 3-core fallbacks -> {args.out_dir}")

    if args.pack:
        stage_npy(args.out_dir)
        extra = [os.path.join(HERE, "export_rk.py"),
                 os.path.join(HERE, "build_rknn.py"),
                 os.path.join(HERE, "src", "layers", "layers.py"),
                 os.path.join(HERE, "src", "models", "image_model.py"),
                 os.path.join(HERE, "src", "models", "video_model.py"),
                 os.path.join(HERE, "build_rknn_i8.py"),
                 os.path.join(HERE, "qat.py")]
        pack(args.pack, args.out_dir, [f for f in extra if os.path.exists(f)])
        print(f"packed -> {args.pack}")


if __name__ == "__main__":
    main()
