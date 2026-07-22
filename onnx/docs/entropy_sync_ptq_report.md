# Cross-Platform Entropy Synchronization: Analysis and PTQ Experiments

Status: completed investigation (2026-07-21)
Scope: `onnx/` codec (ONNX Runtime CPU/GPU execution providers, C rANS entropy coding)

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

### Remaining untried paths to "sync + RD"

1. **INT8 QOperator + better calibration**: percentile clipping and/or
   cross-layer equalization (CLE, folds per-channel range imbalance into
   neighboring weights — data-free, e.g. AIMET) attack the u8 range problem at
   its source while keeping integer kernels. Cheapest next experiment.
2. **Custom fixed-point inference**: the 7 networks are small (1-19 convs);
   implementing them as fixed-order int32-accumulation convs in C makes
   bit-exactness structural and frees the quantization granularity entirely
   (per-channel int16 possible). Medium effort, definitive.
3. **QAT with boundary-avoidance regularization**: requires training code,
   which this repo does not contain.
4. **Tiny checksum reframe**: a 4-byte params_fusion checksum per frame
   (~0.01% bitrate at 1080p) + concealment-on-mismatch is cheaper than every
   option above, if "zero" is relaxed to "near-zero".

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

Constraints honored: no changes to `onnx/src`, `onnx/rans`, or the original
`onnx/models/`; all experiments CPU-only via `.venv`.
