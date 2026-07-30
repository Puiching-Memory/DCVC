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

Place `.rknn` siblings from `dcvc_rk_1080p_i8.zip` plus runtime tables:

| file | role |
|------|------|
| `*.rknn` | 21 subnets (intra + inter + AR) |
| `bitest_*.npy` / `gaussian_*.npy` | z/y CDF |
| `q_scale_enc/dec.npy` | intra analysis/synthesis QP |
| `q_encoder/decoder/feature/recon.npy` | inter QP banks |

## Run e2e

```bash
# powers on NPU then runs I + P
bash rknn/scripts/run_e2e.sh rknn/models/1080p_i8 1088 1920 32 3
```

Frame 0 = I; remaining = UF-LD P. Reports wall ms, NPU `PERF_RUN` ms, bpp, PSNR,
plus a stage table (`wall / npu / set / get / cpu`). `set`/`get` are FP32
`rknn_inputs_set` / `rknn_outputs_get`. Set `DCVC_PROFILE=0` to suppress tables.

Placeholder CDF/QP banks (`scripts/gen_placeholder_assets.py`) are enough for **NPU timing**; replace with real export tables for meaningful PSNR/bpp.
