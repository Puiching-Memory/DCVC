#!/usr/bin/env python3
"""Benchmark an I+P frame sequence (DCVC CPU codec) on a real video, comparing
all-I-frame vs I+P-frame coding at the same resolution/QP.

For I+P: frame 0 uses the intra codec; frames 1..N-1 use the inter codec,
each referencing the previous reconstruction (closed loop, bit-exact enc/dec).
For all-I: every frame uses the intra codec independently.

Reports per-frame PSNR, bytes, and total rate.
"""
import argparse, os, struct, subprocess, sys, tempfile, time
import numpy as np
from PIL import Image


def extract_frame(video, index, w, h, tmp):
    png = os.path.join(tmp, f"f{index:05d}.png")
    subprocess.run(["ffmpeg","-v","error","-y","-i",video,
        "-vf",f"select=eq(n\\,{index}),scale={w}:{h}:flags=lanczos",
        "-vsync","0","-frames:v","1","-pix_fmt","rgb24",png], check=True)
    arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
    nchw = (arr.astype(np.float32)/255.0).transpose(2,0,1)[None]
    return nchw, png

def save_npy(path, nchw):
    n,c,hh,ww = nchw.shape
    hdr = f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {c}, {hh}, {ww}), }}"
    pad = (64 - ((len(hdr)+10)%64))%64; hdr += " "*pad
    with open(path,"wb") as f:
        f.write(b"\x93NUMPY"); f.write(struct.pack("<BBH",1,0,len(hdr)))
        f.write(hdr.encode()); f.write(nchw.astype("<f4").tobytes())

def load_npy(path):
    with open(path,"rb") as f:
        assert f.read(6)==b"\x93NUMPY"; _,_,hl=struct.unpack("<BBH",f.read(4)); f.read(hl)
        return np.frombuffer(f.read(),"<f4").copy()

def psnr(a,b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 999.0 if mse<=0 else 10*np.log10(1.0/mse)

def run_i(intra_exe, model_dir, tmp, idx, x_npy, qp, H, W):
    binp = os.path.join(tmp, f"i_{idx:05d}.bin")
    for ext in ("", ".enc.npy"):
        if os.path.exists(binp+ext): os.remove(binp+ext)
    r = subprocess.run([intra_exe,"--model-dir",model_dir,"--encode",binp,
                        str(H),str(W),str(qp),x_npy],
                       capture_output=True,text=True)
    if r.returncode: sys.stderr.write(r.stdout+r.stderr); raise RuntimeError("i encode")
    rec = load_npy(binp+".enc.npy").reshape(1,3,H,W)
    return rec, os.path.getsize(binp)

def run_p(inter_one_exe, model_dir, tmp, idx, ref_npy, x_npy, qp, H, W):
    binp = os.path.join(tmp, f"p_{idx:05d}.bin")
    for ext in ("",):
        if os.path.exists(binp+ext): os.remove(binp+ext)
    env = dict(os.environ)
    env["DCVC_DUMP_XHAT"] = binp+".enc.npy"
    env["DCVC_DUMP_YHAT"] = os.devnull
    env["DCVC_DUMP_FEAT"] = os.devnull
    r = subprocess.run([inter_one_exe, ref_npy, x_npy, str(qp), model_dir, binp],
                       capture_output=True,text=True, env=env)
    if r.returncode: sys.stderr.write(r.stdout+r.stderr); raise RuntimeError("p encode")
    rec = load_npy(binp+".enc.npy").reshape(1,3,H,W)
    return rec, os.path.getsize(binp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--intra-exe", required=True)
    ap.add_argument("--inter-one-exe", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--qp-i", type=int, default=32)
    ap.add_argument("--qp-p", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--mode", choices=["ip","all_i","both"], default="both")
    args = ap.parse_args()
    assert args.width%64==0 and args.height%64==0
    pix = args.width*args.height
    tmp = tempfile.mkdtemp(prefix="dcvc_seq_")

    idxs = [args.start+i for i in range(args.frames)]
    print(f"video={os.path.basename(args.video)} frames={idxs} "
          f"res={args.width}x{args.height} qp_i={args.qp_i} qp_p={args.qp_p}")
    xs = []
    for idx in idxs:
        nchw,_ = extract_frame(args.video, idx, args.width, args.height, tmp)
        xs.append(nchw)

    def run_sequence(mode):
        total=0; ps=[]
        print(f"\n=== {mode.upper()} ===")
        print(f"{'frame':>5} {'type':>4} {'bytes':>8} {'bpp':>7} {'PSNR':>8}")
        ref=None
        for k,idx in enumerate(idxs):
            xnpy=os.path.join(tmp,f"x_{idx}.npy"); save_npy(xnpy, xs[k])
            if mode=="ip" and k>0:
                refnpy=os.path.join(tmp,f"ref_{idx}.npy"); save_npy(refnpy, ref)
                rec,nb=run_p(args.inter_one_exe,args.model_dir,tmp,idx,refnpy,xnpy,args.qp_p,args.height,args.width)
                ft="P"
            else:
                rec,nb=run_i(args.intra_exe,args.model_dir,tmp,idx,xnpy,args.qp_i,args.height,args.width)
                ft="I"
            p=psnr(xs[k],rec); ps.append(p); total+=nb; ref=rec
            print(f"{idx:5d} {ft:>4} {nb:8d} {nb*8/pix:7.3f} {p:8.2f}")
        print(f"--- {mode}: total={total} B ({total/1024:.1f} KB), "
              f"{total/len(idxs):.0f} B/frame, mean PSNR={np.mean(ps):.2f} dB")
        return total, np.mean(ps)

    res={}
    if args.mode in ("all_i","both"): res["all_i"]=run_sequence("all_i")
    if args.mode in ("ip","both"):    res["ip"]=run_sequence("ip")
    if len(res)==2:
        ti,pi=res["all_i"]; tp,pp=res["ip"]
        print(f"\n=== SUMMARY ===")
        print(f"all-I: {ti/1024:.1f} KB, {ti/len(idxs):.0f} B/frame, {pi:.2f} dB")
        print(f"I+P  : {tp/1024:.1f} KB, {tp/len(idxs):.0f} B/frame, {pp:.2f} dB")
        print(f"rate saving: {(1-tp/ti)*100:.1f}%   PSNR delta: {pp-pi:+.2f} dB")

if __name__=="__main__":
    main()
