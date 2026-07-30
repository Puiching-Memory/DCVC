#!/usr/bin/env python3
"""Micro-bench: does splitting the 512->256 concat-adaptor restore 3-core?

Model A (current):  cat([a,b]) -> Conv(512->256,1)            [the fallback layer]
Model B (proposed): Conv(256->256,1)(a) + Conv(256->256,1)(b)  [split, GLU-style]
Both at the REAL inter_encoder adaptor resolution: a,b = (1,256,136,240).
Weights transfer 1:1: W[:, :256] -> Wa, W[:, 256:] -> Wb.
"""
import os, sys, tempfile, re
os.environ["RKNN_TILE_ANALYSIS"] = "1"
import torch, onnx
from torch import nn
from rknn.api import RKNN

TARGET = "rk3588"
FB = re.compile(r"Failed to config layer: '(.*?)' using 3Core")
TILE = re.compile(r"TileChannel|TC_TILING.*adaptor|TC_COST.*adaptor|ic=512|ic=256 oc=256", re.I)

class AdaptorCat(nn.Module):     # A: current
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 256, 1)
    def forward(self, a, b):
        return self.conv(torch.cat((a, b), dim=1))

class AdaptorSplit(nn.Module):   # B: proposed
    def __init__(self):
        super().__init__()
        self.ca = nn.Conv2d(256, 256, 1)
        self.cb = nn.Conv2d(256, 256, 1)
    def forward(self, a, b):
        return self.ca(a) + self.cb(b)

def build(name, net, args):
    net.eval()
    onx = tempfile.mkstemp(suffix=".onnx", prefix=f"_mb_{name}_")[1]
    torch.onnx.export(net, args, onx, dynamo=False, opset_version=17,
                      input_names=[f"in{i}" for i in range(len(args))], output_names=["out0"])
    m = onnx.load(onx)
    for inp in m.graph.input:                       # bake concrete
        for d in inp.type.tensor_type.shape.dim:
            d.ClearField("dim_param")
    # set real shapes
    for inp, shp in zip(m.graph.input, [(1,256,136,240)]*len(args)):
        for d, v in zip(inp.type.tensor_type.shape.dim, shp):
            d.dim_value = v
    onnx.save(m, onx)
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_mblog_{name}_")[1]
    so, se = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR|os.O_CREAT|os.O_TRUNC)
    r = RKNN(verbose=True); rc=None
    try:
        os.dup2(fd,1); os.dup2(fd,2)
        r.config(mean_values=[], std_values=[], target_platform=TARGET, float_dtype="float16")
        rc = r.load_onnx(model=onx)
        if rc==0: rc = r.build(do_quantization=False)
    finally:
        os.dup2(so,1); os.dup2(se,2); os.close(fd); os.close(so); os.close(se); r.release()
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(logf, errors="ignore").read())
    fb = FB.findall(t)
    print(f"\n===== {name}  rc={rc}  3-core-fallbacks={len(fb)} =====")
    for f in fb: print(f"   [FB] {f}")
    # show TC_TILING for the conv(s)
    seen=set()
    for ln in t.splitlines():
        if "TC_TILING op=Conv" in ln:
            short = ln.split("TC_TILING ")[1][:120]
            if short not in seen:
                seen.add(short); print(f"   TIL {short}")
    # show all TC_COST tiles=1 lines (the chosen baseline cost)
    for ln in t.splitlines():
        if "TC_COST tiles=1 " in ln and ln not in seen:
            seen.add(ln); print(f"   CST {ln.split('] ')[-1][:140]}")
    return len(fb)

a = torch.randn(1,256,136,240); b = torch.randn(1,256,136,240)
fA = build("A_cat_512to256",  AdaptorCat(),   (a, b))
fB = build("B_split_2x256to256", AdaptorSplit(), (a, b))
print(f"\n>>> RESULT: cat-adaptor fallbacks={fA}  | split-adaptor fallbacks={fB}")
print(f">>> {'3-core RESTORED by split' if (fA>0 and fB==0) else 'no improvement / unexpected'}")
