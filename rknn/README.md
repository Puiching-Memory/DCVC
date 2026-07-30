# DCVC-RK C runtime (RK3588 / RKNN)

Standalone C inference pipeline for DCVC-RK INT8 `.rknn` packs.
**Zero dependency on `onnx/`** — ships its own UF-format rANS + npy reader.

## Build

```bash
bash rknn/scripts/build_linux.sh
# -> rknn/out/build/linux-aarch64/test_rknn_e2e
```

Requires `librknnrt` and `rknn_api.h` on the board.

## Model directory

Fused pack (**14** `.rknn`, was 21). From `dcvc_rk_1080p_i8.zip` → `dcvc_rk_1080p_rk3588_i8/`:

| file | role |
|------|------|
| `intra_analysis_hyper.rknn` | analysis + hyper_enc → (y, z) |
| `intra_prior_chain.rknn` | hyper_dec + prior_fusion → params_fusion |
| `intra_synthesis.rknn` | synthesis |
| `y_spatial_prior_*.rknn` | intra AR (4-pass, kept split) |
| `inter_feat_i/p.rknn` | adaptor + extractor → (memory, ctx) |
| `inter_enc_hyper.rknn` | encoder + hyper_enc → (y, z) |
| `inter_prior_chain.rknn` | hyper_dec + temporal + mul/cat + prior_fusion |
| `inter_spatial_prior.rknn` | inter AR (2-pass) |
| `inter_dec_recon.rknn` | decoder + recon → (feature, recon) |
| `bitest_*.npy` / `gaussian_*.npy` | z/y CDF |
| `q_scale_enc/dec.npy`, `q_encoder/decoder/feature/recon.npy` | QP banks |
| `shapes.json` | `"fused": true` + input shapes |

## Run e2e

```bash
# powers on NPU then runs I + P
bash rknn/scripts/run_e2e.sh rknn/models/1080p_i8 1088 1920 32 3
```

Frame 0 = I; remaining = UF-LD P. Reports wall ms, NPU `PERF_RUN` ms, bpp, PSNR,
plus a stage table (`wall / npu / set / get / cpu`).

Env knobs:
- `DCVC_PROFILE=0` — hide stage tables
- `DCVC_RKNN_IO=i8|fp32|i8out|nhwc` — float-hop I/O path for AR / fallback (default **`i8`**)
- `DCVC_RKNN_PIPE_I8=0` — disable pipeline-native INT8 (default **on**: NHWC pass-through in + INT8 out; requant between nets; AR/RGB still float)
- `DCVC_RKNN_DMA=1` — use `rknn_create_mem`/`set_io_mem` (default off; only for `i8`)
- `DCVC_RKNN_ASYNC=0` — disable NPU∥CPU overlap (default on)
- `OMP_NUM_THREADS` — CPU kernels / quant (default 4 when i8/DMA)

Async overlap (default on): `prior_chain` ∥ `z_entropy`; `synthesis`/`dec_recon` ∥
bitstream pack; P-frames also prefetch next-frame preprocess during `dec_recon`.
Stage SUM may exceed frame wall when overlapped.

Placeholder CDF/QP banks are enough for **NPU timing**; replace with real export tables for meaningful PSNR/bpp.
