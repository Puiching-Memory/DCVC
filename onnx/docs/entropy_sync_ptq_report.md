# Cross-Platform Entropy Synchronization: Analysis and PTQ Experiments

Status: completed investigation (2026-07-21); **solution landed** — see
[fxp_cross_platform_rans.md](fxp_cross_platform_rans.md) for the FXP custom-op
design that is now the default `onnx/models/` path.
Scope: `onnx/` codec (ONNX Runtime CPU/GPU execution providers, C rANS entropy coding)

> **Reader note.** Sections 1–7 below are the original problem analysis and
> INT8/INT16 PTQ experiments. Section 8 documents rANS runtime protection.
> The production fix is **fixed-point ORT custom ops + integer scale→index**
> (`fxp_cross_platform_rans.md`), not INT8 QOperator (too costly in RD) or
> INT16 QDQ (not bit-exact on ORT CPU).

## 1. Problem: cross-device FP differences can desync the entropy coder

The y-stream CDF is selected per element by quantizing the NN-predicted `scale`
into an integer index 0..127 (`clamp -> log -> floor`), then looking up a static
integer CDF table (`models/gaussian_cdf.npy`). Decoder and encoder each recompute
this index from their **own** neural-network outputs:

- `src/cpu_ar_codec.c` (`build_index_dec` / `build_index_enc`)
- same design in the original torch layer (`src/layers/cuda_inference.py`,
  `src/models/entropy_models.py` `GaussianEncoder`)

`floor` is discontinuous: a 1-ULP cross-platform difference (different ORT
version, AVX2 vs AVX-512 kernels, CPU vs CUDA EP) can flip an index near a
boundary. One flipped index desynchronizes rANS for the rest of the frame, and
the corrupted `y_hat` feeds the next autoregressive round, amplifying the damage.

The rANS decoder has **no protection** against this:

- `rans/rans.cpp:366` — CDF linear scan `while (cdf[s++] <= cum_freq)` has no
  upper bound; a desynced state scans past the table (heap over-read).
- `rans/rans_byte.h:138-140` — renorm reads `*ptr++` with no end-of-stream check
  (`set_stream` stores no tail pointer), so a desync can walk off the buffer and
  segfault.
- The bitstream has no CRC / sync markers; corruption is undetectable except by
  PSNR after the fact.

Safe paths: z uses a static position-based CDF (`rans.cpp:428`), so z can never
desync; `mean`/`qenc`/`qdec` do not index the CDF and only cause bounded
reconstruction drift (which then propagates through the spatial prior).

### Existing mitigations in the model layer (src/)

- CDF tables are precomputed once as 16-bit fixed-point integers
  (`entropy_models.py:248-283`) — table *contents* are platform-independent.
- scale clamped to [0.11, 16]; z uses position-based static CDF; q-tables are
  static per-qp lookups.
- **No** snapping/hysteresis on the scale index, no deterministic-inference
  switch, no CRC/framing integrity in `utils/stream_helper.py`. The design
  assumes encoder and decoder run the *same* inference stack.

## 2. Enabler: GPU execution providers in `onnx/`

The codec was CPU-only; GPU EP support was added (this is what raised the
cross-device question in the first place):

- `onnx/src/onnx_engine.cpp` — `use_gpu` (0=CPU, 1=CUDA, 2=TensorRT) honored via
  ORT V2 EP APIs, `GetAvailableProviders` gate, graceful CPU fallback,
  `DCVC_USE_GPU` / `DCVC_GPU_DEVICE` env vars, once-per-process logging.
- `onnx/CMakeLists.txt` — `-DDCVC_ORT_GPU=ON` fetches the `*-gpu` ORT package
  and ships the provider libraries.
- Verified: CPU-only package + `DCVC_USE_GPU=1/2` warns and falls back; default
  CPU roundtrips (`test_ar_cpu`, `test_cpu_end2end`, `test_cpu_inter`) PASS
  bit-exact. Real-GPU loading was **not** tested here (no network access to the
  GPU package, no cuDNN on the box).

Note: CPU-encode + GPU-decode (or vice versa) is currently **unsafe** — the
desync risk above applies with near-certainty at 1080p (~630k scale indexes per
frame; cross-EP relative FP error ~1e-5 implies hundreds of boundary flips).

## 3. Zero-extra-bitrate option space

No bitrate overhead means no sync information can be transmitted, leaving three
directions:

1. **Make both sides bit-exact** — lock the inference stack (same ORT build/EP),
   or integer quantization of the entropy-parameter networks (int arithmetic is
   associative → cross-platform bit-exact by construction).
2. **Reduce flip probability** — boundary-avoidance training / discrete-index
   models (needs training code, unavailable in this repo).
3. **Detect + tolerate** — rANS decoder bounds checks (crash → error code),
   statistical desync detection (decoded-symbol distribution diverges within a
   few dozen symbols), error concealment for video.

Experiments below quantify direction 1 without training code (PTQ only).

## 4. Experiment A: INT8 PTQ (QOperator)

7 entropy-parameter networks quantized (static PTQ, u8 activations / s8
per-channel weights, MinMax calibration on 12 dumped samples = 4 frames x 3 qp,
dataflow cross-checked against C dumps at 8e-6):
`hyper_dec`, `y_prior_fusion`, `y_spatial_prior_reduction`,
`y_spatial_prior_adaptor_{1,2,3}`, `y_spatial_prior`.

Key engineering decisions:

- **QOperator format, not QDQ**: QDQ falls back to FP32 conv when graph
  optimizations are disabled → breaks bit-exactness. QOperator bakes
  `QLinearConv` into the graph → integer kernels under every config.
- **Output-conv channel-group split** for `y_prior_fusion` / `y_spatial_prior`
  (bit-exact FP32-equivalent surgery): heterogeneous output channels
  (means ~±20 vs scales 0.02..2.5) otherwise crush the per-tensor u8 range.
  Reduced scale error 10x (0.110 -> 0.011).

Results:

| Check                                                       | Result                                |
| ----------------------------------------------------------- | ------------------------------------- |
| 4-config bit-exact (threads {1,8} x graph-opt {off,all})    | 7/7 PASS (FP32 originals: 6/7 DIFFER) |
| Cross-version (ORT 1.19 C vs ORT 1.27 Python) params_fusion | **diff = 0 (bit-exact)**              |
| C roundtrip (ORT 1.19), enc-dec                             | PASS, maxdiff = 0                     |
| RD cost (3 frames x qp {22,32,42}, 512x512)                 | **bytes x3.21, PSNR -1.89 dB**        |
| RD cost (1920x1024)                                         | bytes x3.39, PSNR -1.48 dB            |

Root cause of the RD cost: ONNX `QLinearConv` only supports **per-tensor**
activations, and u8/MinMax is too coarse for the heterogeneous intermediate
ranges. On highly-compressible content (86% of scales at the 0.11 clamp, very
sharp CDFs), small scale perturbations push residuals into Gaussian tails
(~18 bit/symbol) and the bitrate multiplies. Ablations: quantizing only a
subset still costs +199%; calibration method (entropy vs minmax) made no
difference. This is close to the PTQ floor without QAT.

## 5. Experiment B: INT16 PTQ (QDQ)

Same 7 networks, `QuantType.QInt16` activations (QDQ only — ORT tooling rejects
16-bit QOperator), no conv split, opset 21 / ir 8.

| Check                       | Result                                       |
| --------------------------- | -------------------------------------------- |
| RD cost (512x512)           | **bytes x1.034 (+3.1%), PSNR +0.05 dB**      |
| RD cost (1920x1024)         | bytes x1.056, PSNR +0.05 dB                  |
| 4-config bit-exact          | 7/7 **thread-sensitive** (FP32 conv path)    |
| Cross-version params_fusion | diff = 6.38e-3 (**not** bit-exact)           |
| Runtime cost                | +10% (512x512) .. +49% (1920x1024) wall time |

Mechanism: ORT CPU has **no int16 integer conv kernel** — 16-bit QDQ convs
execute as `DQ -> FP32 Conv -> Q` (confirmed via optimized-graph dump: no
`QLinearConv` nodes). The per-layer requantize *attenuates* cross-platform FP32
noise ~31x (scale error 0.0009..0.0021; index agreement with FP32 98.5-99.3%),
but attenuation is not elimination: two deployments with different thread counts
or CPU ISAs can still desync.

## 6. Verdict

| Route                        | Sync guarantee               | RD cost       | Position                     |
| ---------------------------- | ---------------------------- | ------------- | ---------------------------- |
| INT8 QOperator (+split)      | architectural bit-exact      | x3.2 bitrate  | sync solved, RD unacceptable |
| INT16 QDQ                    | none (31x noise attenuation) | ~free (+3-5%) | RD solved, sync best-effort  |
| Lock stack + detect/tolerate | deployment constraint        | zero          | engineering fallback         |

per-tensor u8 was the RD bottleneck, and INT16 removes it — but ORT CPU executes
16-bit convs in FP32, so the bit-exact property is lost. The two goals are
orthogonal in this toolchain today.

### Remaining paths to "sync + RD"

1. **INT8 QOperator + better calibration**: percentile clipping and/or
   cross-layer equalization (CLE). Scripts exist (`cle_equalize.py`,
   `run_ablation.py`); not yet shown to bring RD back to usable.
2. **Custom fixed-point via ORT custom op (landed)**: domain `com.dcvc` with
   `FxpConv` / `FxpWsRelu` / `FxpConv1x1` (`src/fxp/`). Export all 7 entropy
   nets with `python/fxp_export_entropy_nets.py`. Verified:
   - kernel / ORT repeat bit-exact
   - `test_cpu_end2end models_fxp 128 128 32` → encode↔decode max_diff=0
   - **Integer CDF index path** (`fxp_scale_index.c`): float→Q18→Q12→table
     lookup replaces `logf`/`floorf` in intra+inter enc/dec. ~99.6% agree with
     legacy float-log on a linspace probe; enc/dec share one rule.
   - **Real-content RD** (UVG Beauty/Bosphorus/Jockey, 256², qp 12..52):
     - int8 weights + random calib: mean bitrate **~2.1–2.5×** FP32
     - int16 weights + means/scales split + **random** N(0,1) act calib:
       mean **~2.43×** (Jockey ~3.6×). Root cause: random calib overestimated
       activation absmax by ~4–12× → coarse `x_scale`.
     - **int16 + real-content act calib** (`ptq_dump_calib.py` dumps,
       absmax/`--percentile`): mean bitrate **~1.000×** FP32, |ΔPSNR| ≲ 0.01 dB,
       enc↔dec max_diff=0. Leave-one-out (calib Beauty+Bosphorus, test Jockey)
       still ~1.00×. The fxp nets now ship as the DEFAULT `onnx/models/`
       directory (FP32 moved to `models_fp32/`); results in `python/rd_fxp_real/`.
     Export: `fxp_export_entropy_nets.py --calib-dir … [--cle] [--percentile]`.
   Optional next: broader calib set for production; CLE/percentile were
   prepared but unnecessary once real absmax calib is used.
3. **QAT with boundary-avoidance regularization**: requires training code,
   which this repo does not contain.
4. **Tiny checksum reframe**: a 4-byte params_fusion checksum per frame
   (~0.01% bitrate at 1080p) + concealment-on-mismatch is cheaper than every
   option above, if "zero" is relaxed to "near-zero".

## 8. rANS runtime protection (landed)

The earlier analysis noted the decoder had **no protection**: the CDF linear
scan (`rans.cpp:366`) was unbounded, renorm (`rans_byte.h:138`) read past the
buffer with no end check, and the bitstream carried no integrity check. A
single desync'd index would segfault or heap-overread, undetectable except by
PSNR after the fact. Three layers of runtime protection have now landed in
`onnx/rans` and the codec layer; all are zero-extra-bitrate except the optional
CRC trailer.

**Layer 1 — bounds-checked decoder primitives** (`rans_byte.h`)
- `RansDecInitSafe` / `RansDecAdvanceSafe` take an `end` pointer and return
  `false` on overread instead of the unbounded `*ptr++`.
- A bounded `RansDecGetBitsSafe` (in `rans.cpp`, where `bypass_precision`
  lives) guards the bypass-mode bit reads.
- The CDF scan in `decode_one_symbol` is now capped at `cdf_size` (the last
  entry is `2^SCALE_BITS`, always > any valid `cum_freq`, so hitting the cap
  proves corruption) instead of the old `while (cdf[s++] <= cum_freq)`.

**Layer 2 — deterministic stream-consumption check** (`rans.h` / codecs)
- `RansDecoderLib` now tracks a sticky `_error` flag + `_stream_end` pointer.
  `has_error()` reports any overread/bad-state; `bytes_consumed()` reports
  bytes read since `set_stream()`.
- The rANS state machine is deterministic: a synchronized decode consumes
  **exactly** the bytes the encoder produced. `cpu_ar_codec.c`,
  `cpu_intra_pipeline.c`, and `cpu_inter_pipeline.c` assert
  `bytes_consumed() == stream_size` after decoding — a mismatch proves the
  two sides diverged (e.g. a cross-EP FP difference selected a different CDF),
  even when no explicit overread occurred. On either signal the codec returns
  the new `DCVC_CPU_ERR_ENTROPY` (`"entropy_sync"`) *before* garbage symbols
  enter the autoregressive loop.
- Exposed via the C API: `dcvc_rans_decoder_has_error()`,
  `dcvc_rans_decoder_bytes_consumed()`.

**Layer 3 — optional CRC-32 framing** (`rans_c.h`: `dcvc_crc32`)
- `test_cpu_end2end --encode/--decode` honor `DCVC_CRC=1`: the encoder appends
  a 4-byte CRC-32 over the entropy stream; the decoder verifies it before any
  rANS work, so an in-transit bit flip is caught deterministically (~0.02%
  overhead at 1080p) instead of silently corrupting the reconstruction. The
  rANS state machine alone cannot detect channel errors.

**Verification** (all CPU EP):
- Normal round-trip: `test_ar_cpu`, `test_cpu_end2end 256 256 32` — still
  `max_diff=0` (protection is zero-overhead / no false positives).
- Single-byte flip mid-stream: decode returns `entropy_sync` (exit 1) instead
  of segfaulting.
- Poison streams (truncated / all-zero / garbage, wrapped in a valid `DCV1`
  header): every case exits cleanly with a status code, no crash.
- CRC: `DCVC_CRC=1` passes on a clean stream, reports a CRC mismatch on a
  flipped byte.
- MinGW cross-compiled `.exe` run under Wine: normal round-trip `max_diff=0`,
  single-byte flip → `entropy_sync`, CRC framing OK — i.e. the protection works
  identically in the cross-compiled PE.

  > **Wine caveat (important).** These Wine results are a *binary smoke test*,
  > not cross-platform evidence. Wine is a Windows API compatibility layer, not a
  > CPU emulator — the MinGW `.exe`'s x86-64 instructions run natively on the
  > Linux host, so the Wine round-trip only proves the PE links, the DLL loads,
  > and the protection logic fires. It does **not** exercise the actual
  > cross-platform floating-point risk (MSVC vs GCC, different OS, different CPU
  > / ORT build). The genuine GCC↔MSVC measurement lives in the `max_diff=0`
  > table of the README's *Cross-platform interoperability test*, which requires
  > two physical machines of different OSes.

The protection detects desync/corruption; it does not *repair* it. For video,
pair it with error concealment (reuse the previous frame's reconstruction on
`DCVC_CPU_ERR_ENTROPY`) at the application layer.

## 7. Artifacts and reproduction

- Scripts: `onnx/python/ptq_dump_calib.py`, `ptq_quantize.py`
  (`--activation-type {uint8,int16}`, `--format`, `--no-split`),
  `ptq_verify.py`, `ptq_rd_compare.py`
- Calibration data: `onnx/python/calib_data/` (12 samples + manifest)
- Quantized model dirs: `onnx/models_int8/` (QOperator), `onnx/models_int16/`
  (QDQ int16) — full copies of `models/` with the 7 networks replaced;
  unquantized files byte-identical.
- RD outputs/logs: `onnx/python/rd_out*/`
- Spot checks reproduced by hand:
  `onnx/build/test_cpu_end2end onnx/models_int8 512 512 32 <x.npy> <rec.npy>`
  -> PASS, maxdiff 0, PSNR 39.50 dB; single-model INT16 probe
  (y_spatial_prior_reduction) PSNR 40.75 dB vs FP32 40.72 dB.

Note: section 7 (PTQ experiments) left `onnx/src`, `onnx/rans`, and
`onnx/models/` untouched. Section 8 (runtime protection) is the exception: it
adds bounds checks to `onnx/rans/rans_byte.h`, `rans.h`, `rans.cpp`, `rans_c.*`,
and wires the new `DCVC_CPU_ERR_ENTROPY` status into the codecs.
All experiments CPU-only via `.venv`.
