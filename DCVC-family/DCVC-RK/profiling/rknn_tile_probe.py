#!/usr/bin/env python3
"""Probe a11 TileChannel cost-model on DCVC-RK subnets.

Env (read at rknn import time, set BEFORE launching this process):
  RKNN_TILE_ANALYSIS=1          -> emit TC_TILING/TC_COST decision logs
  RKNN_FORCE_TILE_NUM=<int>     -> force channel tile count
  RKNN_TILE_CHANNEL_SPLIT_VIEW  -> emit split-view layout
  RKNN_TARGET                   -> rk3588 (default)
Usage:
  RKNN_TILE_ANALYSIS=1 python -u rknn_tile_probe.py inter_encoder inter_decoder
  RKNN_FORCE_TILE_NUM=2 RKNN_TILE_ANALYSIS=1 python -u rknn_tile_probe.py inter_encoder
"""
import os, sys, json, tempfile, re

ROOT = "/root/workspace/DCVC"
sys.path.insert(0, ROOT)

# env MUST be set before importing rknn (lib reads them at load). Caller sets them.
import onnx
from rknn.api import RKNN

ONNX_DIR = f"{ROOT}/onnx/models_rk"
TARGET   = os.environ.get("RKNN_TARGET", "rk3588")
manifest = json.load(open(f"{ONNX_DIR}/shapes.json"))
SHAPES   = manifest["subnets"]
FTN      = os.environ.get("RKNN_FORCE_TILE_NUM", "")   # echo for the record
print(f"[probe] TILE_ANALYSIS={os.environ.get('RKNN_TILE_ANALYSIS')} "
      f"FORCE_TILE_NUM={FTN!r} SPLIT_VIEW={os.environ.get('RKNN_TILE_CHANNEL_SPLIT_VIEW')}")

TILE_RE = re.compile(r"TC_|TileChannel|channel tile|Split view|FORCE_TILE|CNA|legal channel tile", re.I)
FB_RE   = re.compile(r"Failed to config layer: '(.*?)' using 3Core")

def bake(name, shapes):
    m = onnx.load(f"{ONNX_DIR}/{name}.onnx")
    for inp, shp in zip(m.graph.input, shapes):
        for dim, val in zip(inp.type.tensor_type.shape.dim, shp):
            dim.ClearField("dim_param"); dim.dim_value = int(val)
    for o in m.opset_import:
        if (o.domain or "ai.onnx") == "ai.onnx" and o.version > 19:
            o.version = 19
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix=f"_tp_{name}_"); os.close(fd)
    onnx.save(m, path); return path

def build_capt(name, shapes):
    baked = bake(name, shapes)
    logf  = tempfile.mkstemp(suffix=".log", prefix=f"_tplog_{name}_")[1]
    so, se = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    r = RKNN(verbose=True); rc = None
    try:
        os.dup2(fd, 1); os.dup2(fd, 2)
        r.config(mean_values=[], std_values=[], target_platform=TARGET, float_dtype="float16")
        rc = r.load_onnx(model=baked)
        if rc == 0:
            rc = r.build(do_quantization=False)
    finally:
        os.dup2(so, 1); os.dup2(se, 2); os.close(fd); os.close(so); os.close(se)
        r.release()
    return rc, logf

def report(name, rc, logf):
    raw = open(logf, errors="ignore").read()
    t = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    fb = FB_RE.findall(t)
    tc_lines = [ln.strip() for ln in t.splitlines() if TILE_RE.search(ln)]
    # de-dup consecutive identical lines (verbose likes to repeat)
    uniq = []
    for ln in tc_lines:
        if not uniq or uniq[-1] != ln:
            uniq.append(ln)
    print(f"\n===== {name}  rc={rc}  fallbacks={len(fb)}  TC-lines={len(uniq)} =====")
    for f in fb:
        print(f"   [3CORE-FB] {f}")
    for ln in uniq:
        print(f"   {ln}")
    return len(fb), len(uniq)

def main():
    names = sys.argv[1:] or ["inter_encoder", "inter_decoder"]
    tot_fb = 0
    for n in names:
        if n not in SHAPES:
            print(f"skip {n}: not in shapes.json"); continue
        rc, logf = build_capt(n, SHAPES[n])
        nfb, _ = report(n, rc, logf)
        tot_fb += nfb
    print(f"\n[probe] total fallbacks across {len(names)} subnet(s) = {tot_fb}")

if __name__ == "__main__":
    main()
