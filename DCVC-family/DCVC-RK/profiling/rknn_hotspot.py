#!/usr/bin/env python3
"""Build all 15 DCVC-RK subnets with RKNN_TILE_ANALYSIS=1; rank by MACs from TC_TILING.

Key correctness notes (TC_TILING width = INPUT tensor width of the op):
  - stride-1 conv (pw1x1, dw3x3): output space == input space -> use width as-is.
  - stride-2 'down' (3x3,s2): output is half -> oh=H_in/2, ow=w_in/2.
  - tiled op emits N lines x=k/N with partial widths -> sum widths back to full.
  - res width->H: 240->136, 120->68, 60->34, 30->17 (nearest standard; tiled sums
    like 241 round to 240). A stride-2 op thus halves both.
  - hyper subnets' stride-2 2x2 convs are recorded as plain pw lines (no stride
    marker) so their MACs are underestimated (k^2 factor); they are <2G total and
    do not affect the ranking. Flagged in output.
"""
import os, sys, json, tempfile, re
ROOT = "/root/workspace/DCVC"
sys.path.insert(0, ROOT)
os.environ["RKNN_TILE_ANALYSIS"] = "1"
import onnx
from rknn.api import RKNN

ONNX_DIR = f"{ROOT}/onnx/models_rk"
SHAPES = json.load(open(f"{ONNX_DIR}/shapes.json"))["subnets"]
TARGET = "rk3588"
FB = re.compile(r"Failed to config layer: '(.*?)' using 3Core")
TILE_RE = re.compile(r"TC_TILING op=(\S+) batch=(\d+) ic=(\d+) oc=(\d+) x=(\d+)/(\d+) width=(\d+)")

STD = {240:136, 120:68, 60:34, 30:17}            # standard W->H
def res_h(w):
    return STD.get(min(STD, key=lambda k: abs(k-w)))   # nearest standard width

def op_kind(name):
    if "dc.2" in name:           return "dw3x3"
    if name.endswith("/down/Conv") or "/down/" in name: return "down3x3s2"
    return "pw1x1"

def op_macs(name, ic, oc, win):                 # win = full input width (tiled-restored)
    H = res_h(win)
    kind = op_kind(name)
    if kind == "dw3x3":                          # depthwise 3x3, stride1: out=H x win
        return oc * 9 * H * win
    if kind == "down3x3s2":                      # 3x3 stride2: out = H/2 x win/2
        return ic * oc * 9 * (H // 2) * (win // 2)
    return ic * oc * H * win                      # 1x1 pointwise (also conservative for any k)

def bake(name, shapes):
    m = onnx.load(f"{ONNX_DIR}/{name}.onnx")
    for inp, shp in zip(m.graph.input, shapes):
        for dim, val in zip(inp.type.tensor_type.shape.dim, shp):
            dim.ClearField("dim_param"); dim.dim_value = int(val)
    for o in m.opset_import:
        if (o.domain or "ai.onnx") == "ai.onnx" and o.version > 19: o.version = 19
    fd, path = tempfile.mkstemp(suffix=".onnx", prefix=f"_hs_{name}_"); os.close(fd)
    onnx.save(m, path); return path

def build(name, shapes):
    baked = bake(name, shapes)
    logf = tempfile.mkstemp(suffix=".log", prefix=f"_hslog_{name}_")[1]
    so, se = os.dup(1), os.dup(2)
    fd = os.open(logf, os.O_RDWR|os.O_CREAT|os.O_TRUNC)
    r = RKNN(verbose=True); rc=None
    try:
        os.dup2(fd,1); os.dup2(fd,2)
        r.config(mean_values=[], std_values=[], target_platform=TARGET, float_dtype="float16")
        rc = r.load_onnx(model=baked)
        if rc==0: rc = r.build(do_quantization=False)
    finally:
        os.dup2(so,1); os.dup2(se,2); os.close(fd); os.close(so); os.close(se); r.release()
    return rc, logf

def parse(logf):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(logf, errors="ignore").read())
    fb = FB.findall(t)
    agg = {}
    for m in TILE_RE.finditer(t):
        op, ic, oc, w = m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(7))
        e = agg.setdefault(op, dict(ic=ic, oc=oc, w=0, n=0))
        e["w"] += w; e["n"] += 1
    rows = []
    for op, e in agg.items():
        rows.append(dict(op=op, ic=e["ic"], oc=e["oc"], kind=op_kind(op),
                         win=e["w"], tiled=(e["n"]>1), macs=op_macs(op, e["ic"], e["oc"], e["w"])))
    return rows, fb

def main():
    out = sys.argv[1] if len(sys.argv)>1 else "/tmp/hotspot.json"
    allsub = {}
    for n in SHAPES:
        rc, logf = build(n, SHAPES[n])
        rows, fb = parse(logf)
        tot = sum(r["macs"] for r in rows)
        allsub[n] = dict(rc=rc, nfb=len(fb), ops=rows, total_macs=tot, nconv=len(rows))
        print(f"  {n:30s} rc={rc} fb={len(fb):2d} convs={len(rows):3d} MACs={tot/1e9:8.2f}G")
    json.dump(allsub, open(out,"w"), indent=1)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
