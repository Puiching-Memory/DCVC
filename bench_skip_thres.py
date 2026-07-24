#!/usr/bin/env python3
"""Benchmark: RD curve, bitrate, and timing for skip_thres + fp16 inference."""
import subprocess, os, json, time, sys
import numpy as np

MODELS_DIR = "onnx/models"  # relative to cwd
BUILD_DIR = "onnx/build"
NPY = os.path.abspath("akiyo_10frames.npy")
LD = f"LD_LIBRARY_PATH={BUILD_DIR}"
EXE = os.path.abspath(f"{BUILD_DIR}/test_cpu_inter")

def encode_decode(qp_i, qp_p, skip_thres, label):
    bin_path = f"/tmp/bench_{label}.bin"
    dec_path = f"/tmp/bench_{label}_dec.npy"
    env = os.environ.copy()
    env["DCVC_SKIP_THRES"] = str(skip_thres) if skip_thres > 0 else ""
    # Encode
    t0 = time.time()
    r = subprocess.run(
        [EXE, "--encode", NPY, bin_path, str(qp_i), str(qp_p)],
        capture_output=True, text=True, cwd=MODELS_DIR, env=env)
    t_enc = time.time() - t0
    # Parse per-frame sizes
    sizes = []
    for line in r.stdout.split('\n'):
        if 'frame' in line and 'bytes' in line:
            parts = line.strip().split()
            for p in parts:
                if p.isdigit():
                    sizes.append(int(p)); break
    total_bytes = os.path.getsize(bin_path) - 24  # subtract header
    # Decode
    t0 = time.time()
    r = subprocess.run(
        [EXE, "--decode", bin_path, dec_path],
        capture_output=True, text=True, cwd=MODELS_DIR, env=env)
    t_dec = time.time() - t0
    # PSNR
    orig = np.load(NPY)
    dec = np.load(dec_path)
    N = min(orig.shape[0], dec.shape[0])
    mse = np.mean((orig[:N].astype(np.float64) - dec[:N].astype(np.float64))**2)
    psnr = 10*np.log10(1.0/mse) if mse > 0 else 99.0
    # BPP (bits per pixel)
    _, _, H, W = orig.shape
    total_pixels = N * H * W * 3
    bpp = total_bytes * 8 / total_pixels
    return {
        'total_bytes': total_bytes, 'bpp': bpp, 'psnr': psnr,
        't_enc': t_enc, 't_dec': t_dec,
        'sizes': sizes,
    }

# === Benchmark sweep ===
print("=" * 90)
print(f"{'QP':>6} {'skip_thres':>10} {'Bytes':>8} {'BPP':>7} {'PSNR':>7} {'Enc(s)':>7} {'Dec(s)':>7} {'config':>15}")
print("=" * 90)

results = []
for qp in [22, 32, 42]:
    for st in [0.0, 0.1, 0.2, 0.3, 0.5]:
        label = f"qp{qp}_st{st}"
        r = encode_decode(qp, qp, st, label)
        r['qp'] = qp; r['skip_thres'] = st
        results.append(r)
        print(f"  {qp:>4} {st:>10.1f} {r['total_bytes']:>8} {r['bpp']:>7.3f} {r['psnr']:>6.2f} "
              f"{r['t_enc']:>7.2f} {r['t_dec']:>7.2f} {'baseline' if st==0 else f'skip={st}':>15}")

print("\n" + "=" * 90)
print("Summary: bitrate savings and quality impact at each QP")
print("=" * 90)
for qp in [22, 32, 42]:
    base = next(r for r in results if r['qp']==qp and r['skip_thres']==0.0)
    print(f"\n  QP={qp} (baseline: {base['total_bytes']} B, {base['psnr']:.2f} dB):")
    for r in results:
        if r['qp']!=qp or r['skip_thres']==0: continue
        saved = (1 - r['total_bytes']/base['total_bytes']) * 100
        psnr_loss = base['psnr'] - r['psnr']
        print(f"    skip={r['skip_thres']:.1f}: {r['total_bytes']} B ({saved:+.1f}%), "
              f"PSNR={r['psnr']:.2f} dB ({psnr_loss:+.2f} dB), "
              f"enc={r['t_enc']:.1f}s, dec={r['t_dec']:.1f}s")

# Save results
with open("/tmp/bench_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /tmp/bench_results.json")
