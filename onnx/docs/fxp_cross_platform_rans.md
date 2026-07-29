# FXP ONNX Custom Ops: Cross-Platform rANS Synchronization

Status: landed (default path)  
Scope: `onnx/` CPU codec — ONNX Runtime + `com.dcvc` fixed-point ops + C rANS  
Related: [entropy_sync_ptq_report.md](entropy_sync_ptq_report.md) (problem analysis + INT8/INT16 PTQ experiments)

## 1. Problem

Encoder and decoder each run neural nets to predict per-element Gaussian
`scale`, then map `scale` → CDF index `0..127`, then call rANS. The index map
is **discontinuous** (historically `clamp → log → floor`; now an integer table
lookup with the same semantics).

A **1-ULP** difference in the predicted `scale` near a bin boundary flips the
index. One flip desynchronizes rANS for the rest of the frame; corrupted
`y_hat` then feeds the autoregressive prior and amplifies the error.

Typical sources of ULP drift with plain FP32 ONNX Conv:

- MSVC vs GCC / different ORT builds
- Intra-op thread count / reduction order
- CPU vs CUDA EP (encode on one, decode on the other)

Empirically (FP32 `models/` / older Windows packages): an MSVC-encoded I+P
sequence (`akiyo`, 10×256², qp 32) failed Linux GCC decode at **frame 3** with
`entropy_sync`. Per-frame lengths matched; bitstream content diverged from
frame 3.

**Not affected:** z-stream (position-based static CDF).  
**Critical path:** y-stream scale → CDF index → rANS (zero error tolerance).

## 2. Solution overview

Do **not** send sync side-info. Make both sides compute the **same integers**:

```text
  float I/O (ORT tensors)
        │
        ▼
  ┌─────────────────────────────────────┐
  │  com.dcvc FXP nets (custom ops)     │  int16 act × int16 w → int64 MAC
  │  FxpConv / FxpWsRelu / …            │  associative → cross-OS bit-exact
  └─────────────────────────────────────┘
        │  float scales / means / …
        ▼
  ┌─────────────────────────────────────┐
  │  fxp_scale_index.c                  │  float → Q18 → Q12 → 128-level table
  │  (shared enc + dec)                 │  no runtime logf / floorf
  └─────────────────────────────────────┘
        │  CDF index 0..127
        ▼
  ┌─────────────────────────────────────┐
  │  rANS (+ bounds / bytes_consumed)   │  detect residual desync as entropy_sync
  └─────────────────────────────────────┘
```

Three layers:

| Layer | Mechanism | Role |
|-------|-----------|------|
| A. FXP custom ops | Integer MAC inside entropy (and inter) nets | Same prior tensors on every OS/CPU |
| B. Integer scale index | Shared Q18→Q12→table rule | Same CDF choice given the same floats |
| C. rANS guards | Bounded scan, `bytes_consumed`, optional CRC | Fail loud if anything still diverges |

## 3. Layer A — `com.dcvc` fixed-point ops

Registered with ORT as domain `com.dcvc` (`src/fxp/ort_custom_ops.cpp`).

| Op | Role |
|----|------|
| `FxpConv` / `FxpConv1x1` | Quantize activations with per-tensor `x_scale` → int16; int16 weights; int64 accumulate; dequant to float |
| `FxpWsRelu` | WSiLU via 65536-entry float LUT (deterministic lookup / lerp) |

Why this kills cross-platform drift:

- Integer multiply-add with a **fixed loop order** is associative; FP32 reduction
  order is not.
- Weights and activation quantizers are baked into the ONNX graph at export
  (`python/fxp_export_entropy_nets.py`, real-content calib).
- Final heterogeneous outputs (means vs scales) use **channel-group splits**
  before the last 1×1 so per-tensor activation scales stay sane.

Default ship path: `onnx/models/` is the FXP pack (intra entropy + inter entropy
+ key inter recon/feature nets). FP32 references live in `models_fp32/` for RD
only — **not** for cross-OS bitstreams.

Parallelism stays inside the ORT ecosystem:
`Ort::KernelContext::ParallelFor` uses the session **intra-op** thread pool
(`DCVC_ORT_INTRA_OP_THREADS`, default `min(8, HW)`). Work is split by output
channel; integer results stay bit-exact across thread counts (no private pool).

## 4. Layer B — integer scale → CDF index

File: `src/fxp/fxp_scale_index.c` (used by intra + inter encode/decode).

```text
scale (float32)
  → Q18 (half-away-from-zero)
  → Q12 (round >> 6)
  → clamp to [0.11, 16] in Q12
  → largest i with scale_table_q12[i] <= s   →  index ∈ [0, 127]
```

Encode packs `(round(symbol) << 8) | index`; decode rebuilds indexes the same
way. Encoder and decoder **must** share this rule (and the same FXP models).

Optional probe (not default): `DCVC_SCALE_FP16=1` IEEE-fp16 round-trips each
scale before indexing. Same setting on both sides stays in sync; mixing fp32
and fp16 index paths fails with `entropy_sync` — illustrating the zero-tolerance
requirement on this boundary.

## 5. Layer C — rANS runtime protection

See also §8 of the PTQ report. Summary:

- Bounded CDF scan and renorm (`rans_byte.h` safe APIs)
- After decode: `bytes_consumed() == stream_size` or sticky `has_error()` →
  `DCVC_CPU_ERR_ENTROPY` / `"entropy_sync"`
- Optional `DCVC_CRC=1` 4-byte trailer

These detect corruption/desync; they do not repair it. Application code can
conceal (e.g. reuse previous frame) on that status.

## 6. What we measured

| Check | Result |
|-------|--------|
| FP32 Windows → Linux (akiyo 10f) | Fail `entropy_sync` at frame 3 |
| FXP Windows → Linux (same content, FXP package) | **DECODE PASS**; bitstream **byte-identical** to Linux FXP encode |
| FXP IntraOp threads 1 / 8 / 16 | Bitstreams identical |
| FXP vs FP32 RD (akiyo 10×256² qp 32) | ~**+1.1%** bytes, ~**−0.07 dB** PSNR |
| I-frame-only FXP entropy (UVG 256², earlier) | ~**1.000×** bitrate, ~0 dB ΔPSNR |

Cost is dominated by CPU FXP kernel time (custom int MAC vs ORT FP32 Conv), not
by bitrate. Further speed work should stay on ORT ParallelFor + less float
traffic between FXP ops — not a private thread pool.

## 7. Reproduce / package

```bash
# Export FXP nets (intra + inter) with real calib
uv run python python/fxp_export_entropy_nets.py \
  --src-dir models_fp32 --out-dir models_fxp --calib-dir python/calib_fxp_all

# Linux round-trip
./out/build/linux-x64/test_cpu_inter --model-dir models_720p \
  --encode akiyo_10frames.npy out.bin 32 32
./out/build/linux-x64/test_cpu_inter --model-dir models_720p \
  --decode out.bin out_dec.npy

# Windows package (MinGW cross)
cmake --build out/build/windows-x64-mingw --target dcvc_package
# or build a versioned archive under out/packages/ with scripts/package.sh
```

Cross-OS: encode on one OS, `--decode` on the other with the **same FXP model
dir**. Do not mix FXP bitstreams with FP32 `models_fp32/` or older non-FXP
Windows zips.

## 8. One-line summary

**Cross-platform FP drift flipped rANS CDF indexes.**  
The ONNX `com.dcvc` FXP path replaces entropy (and related) nets with
**integer MAC** and replaces runtime `log/floor` indexing with a **shared
integer scale table**, so encoder and decoder compute identical indexes on
every OS/CPU; rANS guards only catch residual failures.
