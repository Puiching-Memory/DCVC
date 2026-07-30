#!/usr/bin/env python3
"""Generate runtime CDF + QP banks for DCVC-RK C e2e (placeholder tables).

Valid rANS CDFs (precision 16) so encode/decode does not crash. QP banks are
ones — sufficient for NPU wall-time measurement; RD is not meaningful.
"""
import os
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "rknn/models/1080p_i8"
PREC = 16
TOTAL = 1 << PREC  # 65536

def make_cdf(n_sym=256):
    """Uniform-ish CDF of length n_sym+1 ending at TOTAL."""
    # n_sym symbols (0..n_sym-1) + terminal; freq at least 1 each
    freqs = np.full(n_sym, TOTAL // n_sym, dtype=np.int64)
    freqs[: TOTAL % n_sym] += 1
    cdf = np.zeros(n_sym + 1, dtype=np.int32)
    acc = 0
    for i in range(n_sym):
        cdf[i] = acc
        acc += int(freqs[i])
    cdf[n_sym] = TOTAL
    return cdf

def save_bank(name, n_cdfs, n_sym=256):
    per = n_sym + 1
    cdfs = np.zeros((n_cdfs, per), dtype=np.int32)
    base = make_cdf(n_sym)
    for i in range(n_cdfs):
        cdfs[i] = base
    lengths = np.full(n_cdfs, per, dtype=np.int32)
    offsets = np.zeros(n_cdfs, dtype=np.int32)
    np.save(os.path.join(OUT, f"{name}_cdf.npy"), cdfs)
    np.save(os.path.join(OUT, f"{name}_cdf_length.npy"), lengths)
    np.save(os.path.join(OUT, f"{name}_offset.npy"), offsets)
    print(f"{name}: cdfs={cdfs.shape}")

os.makedirs(OUT, exist_ok=True)

# y Gaussian: 128 scale levels, ~256 symbols (zigzag covers int8 range)
save_bank("gaussian", 128, n_sym=256)

# z BitEstimator: need >= qp_max * Z_CH rows. Use 64 qp * 128 = 8192.
save_bank("bitest", 64 * 128, n_sym=256)

# QP banks (ones)
qp_num = 64
np.save(os.path.join(OUT, "q_scale_enc.npy"), np.ones((qp_num, 256), dtype=np.float32))
np.save(os.path.join(OUT, "q_scale_dec.npy"), np.ones((qp_num, 256), dtype=np.float32))
np.save(os.path.join(OUT, "q_encoder.npy"), np.ones((qp_num, 256), dtype=np.float32))
np.save(os.path.join(OUT, "q_decoder.npy"), np.ones((qp_num, 256), dtype=np.float32))
np.save(os.path.join(OUT, "q_feature.npy"), np.ones((qp_num, 256), dtype=np.float32))
np.save(os.path.join(OUT, "q_recon.npy"), np.ones((qp_num, 320), dtype=np.float32))
print("q banks: ones @ qp_num=64")
print("wrote", OUT)
