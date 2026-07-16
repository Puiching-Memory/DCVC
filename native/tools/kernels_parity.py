#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Batch parity test: ATen-free CUDA kernels (dcvc_kernels.cu) vs PyTorch reference.

Loads libdcvc_kernels.so via ctypes, calls each kernel on deterministic FP16
inputs, and compares against the PyTorch fallback semantics from cuda_inference.py.
"""
from __future__ import annotations
import ctypes
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SO = ROOT / "native/build/plugin_demo/libdcvc_kernels.so"
lib = ctypes.CDLL(str(SO))
DEV = "cuda:0"
STREAM = torch.cuda.current_stream(DEV).cuda_stream
FP16 = np.float16

def check(name, got, ref, atol=1e-3):
    err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    ok = err < atol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: max_abs_err={err:.3e}")
    return ok

allok = True

# ---- helpers ----
def rand16(*shape):
    return (np.random.default_rng(sum(shape)).standard_normal(shape) * 2).astype(FP16)

def tensor(arr):
    t = torch.from_numpy(arr).to(DEV)
    return t

def ptr(t):
    return ctypes.c_void_p(t.data_ptr())

# ===========================================================================
print("=== ATen-free kernel parity (11 ops) ===\n")

# 1. round_and_to_int8
print("[round_and_to_int8]")
z = rand16(1, 128, 16, 16)
zt = tensor(z)
z_hat = torch.empty_like(zt)
z_int8 = torch.empty(zt.shape, dtype=torch.int8, device=DEV)
lib.dcvc_k_round_to_int8(ptr(zt), ptr(z_hat), ptr(z_int8),
                         ctypes.c_int(zt.numel()), ctypes.c_void_p(STREAM))
ref_hat = torch.clamp(torch.round(zt), -128., 127.)
allok &= check("z_hat", z_hat.cpu().numpy(), ref_hat.cpu().numpy())
allok &= check("z_int8", z_int8.cpu().numpy().astype(np.int16), ref_hat.cpu().numpy().astype(np.int16))

# 2. add_and_multiply
print("[add_and_multiply]")
x0, x1, q = rand16(1,64,32,32), rand16(1,64,32,32), rand16(1,64,32,32)
x0t, x1t, qt = tensor(x0.copy()), tensor(x1), tensor(q)
lib.dcvc_k_add_and_multiply(ptr(x0t), ptr(x1t), ptr(qt),
                            ctypes.c_int(x0t.numel()), ctypes.c_void_p(STREAM))
ref = ((tensor(x0).float() + tensor(x1).float()) * tensor(q).float()).half()
allok &= check("out", x0t.cpu().numpy(), ref.cpu().numpy())

# 3. clamp_reciprocal_with_quant
print("[clamp_reciprocal_with_quant]")
qd, y = np.abs(rand16(1,64,16,16))+0.1, rand16(1,64,16,16)
qdt, yt = tensor(qd), tensor(y)
qd_c = torch.empty_like(qdt)
y_out = torch.empty_like(yt)
MINV = 0.5
lib.dcvc_k_clamp_recip_quant(ptr(qdt), ptr(yt), ptr(qd_c), ptr(y_out),
                             ctypes.c_float(MINV), ctypes.c_int(qdt.numel()), ctypes.c_void_p(STREAM))
ref_qc = torch.clamp(qdt.float(), min=MINV)
ref_y = yt.float() * (1.0 / ref_qc)
allok &= check("q_dec_clamp", qd_c.cpu().numpy(), ref_qc.cpu().numpy())
allok &= check("y_out", y_out.cpu().numpy(), ref_y.cpu().numpy())

# 4. combine_for_reading_2x
print("[combine_for_reading_2x]")
C2, H, W = 128, 16, 16
x = rand16(1, C2, H, W)
mask = (np.random.default_rng(99).random((1, C2, H, W)) > 0.3).astype(FP16)
xt, mt = tensor(x), tensor(mask)
out = torch.empty((1, C2//2, H, W), dtype=torch.float16, device=DEV)
N = (C2//2) * H * W
lib.dcvc_k_combine_read_2x(ptr(xt), ptr(mt), ptr(out),
                           ctypes.c_int(N), ctypes.c_void_p(STREAM))
ref = (xt.float() * mt.float())
ref = (ref[:, :C2//2] + ref[:, C2//2:]).half()
allok &= check("out", out.cpu().numpy(), ref.cpu().numpy())

# 5. restore_y_2x
print("[restore_y_2x]")
Cy = 64
y = rand16(1, Cy, H, W)
means = rand16(1, Cy*2, H, W)
mask = (np.random.default_rng(77).random((1, Cy*2, H, W)) > 0.2).astype(FP16)
yt, mt, meanst = tensor(y), tensor(mask), tensor(means)
out = torch.empty((1, Cy*2, H, W), dtype=torch.float16, device=DEV)
CyHW = Cy * H * W
lib.dcvc_k_restore_y_2x(ptr(yt), ptr(meanst), ptr(mt), ptr(out),
                        ctypes.c_int(CyHW), ctypes.c_int(CyHW*2), ctypes.c_void_p(STREAM))
yb = torch.cat((yt.float(), yt.float()), dim=1)
ref = ((yb + meanst.float()) * mt.float()).half()
allok &= check("out", out.cpu().numpy(), ref.cpu().numpy())

# 6. restore_y_4x
print("[restore_y_4x]")
means4 = rand16(1, Cy*4, H, W)
mask4 = (np.random.default_rng(88).random((1, Cy*4, H, W)) > 0.2).astype(FP16)
meanst4, mt4 = tensor(means4), tensor(mask4)
out4 = torch.empty((1, Cy*4, H, W), dtype=torch.float16, device=DEV)
lib.dcvc_k_restore_y_4x(ptr(yt), ptr(meanst4), ptr(mt4), ptr(out4),
                        ctypes.c_int(CyHW), ctypes.c_int(CyHW*4), ctypes.c_void_p(STREAM))
yb4 = torch.cat((yt.float(), yt.float(), yt.float(), yt.float()), dim=1)
ref4 = ((yb4 + meanst4.float()) * mt4.float()).half()
allok &= check("out", out4.cpu().numpy(), ref4.cpu().numpy())

# 7. build_index_dec
print("[build_index_dec]")
scales = np.abs(rand16(1, 64, 16, 16)) + 0.01
sct = tensor(scales)
out_u8 = torch.empty(sct.shape, dtype=torch.uint8, device=DEV)
SMIN, SMAX, LMIN, LSTEP = 0.01, 100.0, math.log(0.01), 1.0 / (math.log(100.0/0.01) / 255.0)
lib.dcvc_k_build_index_dec(ptr(sct), ptr(out_u8),
                           ctypes.c_float(SMIN), ctypes.c_float(SMAX),
                           ctypes.c_float(LMIN), ctypes.c_float(LSTEP),
                           ctypes.c_int(sct.numel()), ctypes.c_void_p(STREAM))
sc_c = torch.clamp(sct.float(), SMIN, SMAX)
ref_idx = ((torch.log(sc_c) - LMIN) * LSTEP).round().to(torch.uint8)
allok &= check("index", out_u8.cpu().numpy().astype(np.float32), ref_idx.cpu().numpy().astype(np.float32))

# 8. build_index_enc
print("[build_index_enc]")
symbols = (np.random.default_rng(55).integers(-10, 10, (1,64,16,16))).astype(FP16)
symt = tensor(symbols)
out_i16 = torch.empty(symt.shape, dtype=torch.int16, device=DEV)
lib.dcvc_k_build_index_enc(ptr(symt), ptr(sct), ptr(out_i16),
                           ctypes.c_float(SMIN), ctypes.c_float(SMAX),
                           ctypes.c_float(LMIN), ctypes.c_float(LSTEP),
                           ctypes.c_int(sct.numel()), ctypes.c_void_p(STREAM))
sym_int = np.rint(symbols.astype(np.float32)).astype(np.int32)
ref_enc = ((sym_int << 8) + ref_idx.cpu().numpy().astype(np.int32)).astype(np.int16)
allok &= check("enc", out_i16.cpu().numpy().astype(np.int32), ref_enc.astype(np.int32), atol=1.0)

# 9. process_with_mask
print("[process_with_mask]")
y = rand16(1, 128, 16, 16)
sc = np.abs(rand16(1, 128, 16, 16))
me = rand16(1, 128, 16, 16)
mk = (np.random.default_rng(33).random((1,128,16,16)) > 0.2).astype(FP16)
yt, sct2, met, mkt = tensor(y), tensor(sc), tensor(me), tensor(mk)
yhat = torch.empty_like(yt)
lib.dcvc_k_process_mask(ptr(yt), ptr(sct2), ptr(met), ptr(mkt), ptr(yhat),
                        ctypes.c_float(-1.0), ctypes.c_int(yt.numel()), ctypes.c_void_p(STREAM))
s_hat = sct2.float() * mkt.float()
means_hat = met.float() * mkt.float()
y_q = torch.round((yt.float() - means_hat) * mkt.float())
y_q = torch.clamp(y_q, -128., 127.)
ref_hat = y_q + means_hat
allok &= check("y_hat", yhat.cpu().numpy(), ref_hat.cpu().numpy())

# 10. bias_pixel_shuffle_8
print("[bias_pixel_shuffle_8]")
Hi, Wi = 8, 8
Cin = 192  # 3*64
xb = rand16(1, Cin, Hi, Wi)
biasb = rand16(Cin)
xbt, bt = tensor(xb), tensor(biasb)
outb = torch.empty((1, 3, Hi*8, Wi*8), dtype=torch.float16, device=DEV)
lib.dcvc_k_bias_pixel_shuffle_8(ptr(xbt), ptr(bt), ptr(outb),
                                ctypes.c_int(Hi), ctypes.c_int(Wi), ctypes.c_int(Cin),
                                ctypes.c_int(1), ctypes.c_void_p(STREAM))
import torch.nn.functional as F
ref_b = (xbt.float() + bt.float()[None,:,None,None]).clamp(0,1)
ref_b = F.pixel_shuffle(ref_b, 8)
allok &= check("out", outb.cpu().numpy(), ref_b.cpu().numpy())

# 11. replicate_pad
print("[replicate_pad]")
Hp, Wp = 10, 10
xr = rand16(1, 3, Hi, Wi)
xrt = tensor(xr)
outr = torch.empty((1, 3, Hp, Wp), dtype=torch.float16, device=DEV)
lib.dcvc_k_replicate_pad(ptr(xrt), ptr(outr),
                         ctypes.c_int(3), ctypes.c_int(Hi), ctypes.c_int(Wi),
                         ctypes.c_int(Hp), ctypes.c_int(Wp), ctypes.c_void_p(STREAM))
ref_r = F.pad(xrt.float(), (0, Wp-Wi, 0, Hp-Hi), mode="replicate")
allok &= check("out", outr.cpu().numpy(), ref_r.cpu().numpy())

# ===========================================================================
print(f"\n{'='*50}")
print(f"ALL {'PASS' if allok else 'FAIL'}")
sys.exit(0 if allok else 1)
