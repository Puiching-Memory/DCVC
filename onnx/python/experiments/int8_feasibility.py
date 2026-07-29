#!/usr/bin/env python3
"""INT8 per-channel vs per-tensor activation quantization SNR probe."""
from __future__ import annotations
import argparse, os, glob
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.normpath(os.path.join(REPO, '..'))

def snr_db(signal, noise):
    s = np.mean(signal.astype(np.float64)**2)
    n = np.mean(noise.astype(np.float64)**2)
    return 99.0 if n < 1e-30 else 10*np.log10(s/n)

def q_error(x, mode):
    c = x.shape[0]
    if mode == 'per-channel':
        m = np.max(np.abs(x).reshape(c,-1), axis=1); m[m<1e-12]=1.0
        sc = (m/127.0).astype(np.float32).reshape(-1,*([1]*(x.ndim-1)))
        xq = np.clip(np.rint(x/sc),-128,127)
        dq = xq.astype(np.float32)*sc
    else:
        m = max(float(np.abs(x).max()),1e-12); sc = np.float32(m/127.0)
        xq = np.clip(np.rint(x/sc),-128,127); dq = xq.astype(np.float32)*sc
    return dq, x.astype(np.float32)-dq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calib-dir', default=os.path.join(REPO, 'calib_inter'))
    args = ap.parse_args()
    print('INT8 activation quantization SNR probe (per-channel vs per-tensor)')
    print(f'  calib: {args.calib_dir}')
    print()
    hdr = f"{'net':<28} {'cin':>5} {'shape':<14} {'ch_ratio':>8} {'PC-SNR':>8} {'PT-SNR':>8} {'gain':>7}"
    print(hdr); print('-'*len(hdr))
    pcs, pts = [], []
    for net_dir in sorted(glob.glob(os.path.join(args.calib_dir, '*'))):
        if not os.path.isdir(net_dir): continue
        net = os.path.basename(net_dir)
        dumps = sorted(glob.glob(os.path.join(net_dir, '*.npy')))
        if not dumps: continue
        inp = np.load(dumps[0]).astype(np.float32)
        if inp.ndim==3: inp = inp[None]
        c = inp.shape[1]
        if c < 2: continue
        ch_max = np.max(np.abs(inp).reshape(inp.shape[0],c,-1)[0],axis=1)
        nz = ch_max[ch_max>1e-9]
        ratio = nz.max()/max(nz.min(),1e-9) if len(nz) else 0
        _,pc_err = q_error(inp,'per-channel'); _,pt_err = q_error(inp,'per-tensor')
        pc_s=snr_db(inp,pc_err); pt_s=snr_db(inp,pt_err)
        pcs.append(pc_s); pts.append(pt_s)
        print(f'{net:<28} {c:>5} {str(list(inp.shape[1:])):<14} {ratio:>7.1f}x {pc_s:>7.1f}dB {pt_s:>7.1f}dB {pc_s-pt_s:>+6.1f}dB')
    print('-'*len(hdr))
    if pcs:
        print(f"{'MEAN':<28} {'':>5} {'':<14} {'':>8} {np.mean(pcs):>7.1f}dB {np.mean(pts):>7.1f}dB {np.mean(pcs)-np.mean(pts):>+6.1f}dB")
        print(f'Per-channel is {np.mean(pcs)-np.mean(pts):.1f}dB better than per-tensor on average.')

if __name__=='__main__': main()
