# DCVC-RT Operator Inventory (Phase 0)

> **状态（2026-07-16）**：全链路完成 — intra pipeline 100% bit-exact，ctest 11/11 PASS。image→encode→decode 100% bit-exact。encode→decode round-trip 100% bit-exact。`test_closed_loop` 证明 encode→rANS→decode 在 4× AR 全链路上 100% bit-exact。编码侧 kernel `dcvc_k_process_mask_yq` 已加入 `dcvc_kernels.cu`。

Maps PyTorch / fused CUDA ops to TensorRT or custom Plugin destinations.

## Legend

- **TRT**: standard TensorRT / ONNX op (conv, concat, add, mul, pixel_shuffle via reshape+transpose)
- **Plugin**: custom TensorRT Plugin (from `src/layers/extensions/inference`)
- **Host**: C runtime orchestration (AR loops, DPB, QP gather, entropy)
- **CPU**: rANS entropy on CPU

## Intra (DMCI) subgraphs

| Subgraph               | Key ops                                                               | Destination               | Notes                  |
| ---------------------- | --------------------------------------------------------------------- | ------------------------- | ---------------------- |
| `intra_analysis`       | pixel_unshuffle(8), DepthConvBlock stack, Conv2d downsample, QP scale | TRT + DepthConv Plugin    | Input `[1,3,H,W]` FP16 |
| `intra_hyper`          | hyper_encoder / hyper_decoder ResidualBlocks                          | TRT                       | z at H/64,W/64         |
| `intra_y_prior_fusion` | y_prior_fusion                                                        | TRT                       |                        |
| `intra_spatial_prior`  | y_spatial_prior + adaptors (×4 AR passes)                             | TRT short engine / Plugin | Host loops 4×          |
| `intra_synthesis`      | decoder + bias_pixel_shuffle_8                                        | TRT + Plugin              |                        |
| entropy y/z            | process_with_mask, build_index_*, rANS                                | Plugin + CPU              | 4-pass AR              |

## Inter (DMC) subgraphs

| Subgraph               | Key ops                                              | Destination            | Notes                 |
| ---------------------- | ---------------------------------------------------- | ---------------------- | --------------------- |
| `inter_feature`        | feature_adaptor_i/p, feature_extractor               | TRT + DepthConv Plugin | Uses DPB ref          |
| `inter_analysis`       | encoder (pixel_unshuffle, fused conv1+adaptor), down | TRT + Plugin           | y 128ch               |
| `inter_hyper`          | hyper_enc/dec + temporal_prior                       | TRT                    |                       |
| `inter_y_prior_fusion` | y_prior_fusion                                       | TRT                    |                       |
| `inter_spatial_prior`  | y_spatial_prior (2 AR passes)                        | TRT short engine       | Host loops 2×         |
| `inter_synthesis`      | decoder + recon_generation_net                       | TRT + Plugin           | Stores feature in DPB |

## Fused CUDA → Plugin map

| CUDA / proxy                       | Plugin name             | Priority |
| ---------------------------------- | ----------------------- | -------- |
| `DepthConvProxy`                   | `DcvcDepthConv`         | P0       |
| `SubpelConv2xProxy`                | `DcvcSubpelConv2x`      | P0       |
| `process_with_mask_cuda`           | `DcvcProcessWithMask`   | P0       |
| `combine_for_reading_2x_cuda`      | `DcvcCombineRead2x`     | P0       |
| `restore_y_2x/4x_cuda`             | `DcvcRestoreY`          | P0       |
| `build_index_enc/dec_cuda`         | `DcvcBuildIndex`        | P0       |
| `round_and_to_int8_cuda`           | `DcvcRoundInt8`         | P1       |
| `clamp_reciprocal_with_quant_cuda` | `DcvcClampRecipQuant`   | P1       |
| `add_and_multiply_cuda`            | `DcvcAddMul`            | P1       |
| `bias_pixel_shuffle_8_cuda`        | `DcvcBiasPixelShuffle8` | P0       |
| `replicate_pad_cuda`               | Host pad (or Plugin)    | P1       |
| `bias_wsilu_depthwise_conv2d_cuda` | folded into DepthConv   | P0       |

## ONNX-exportable (slice only)

- `nn.Conv2d`, `nn.ConvTranspose2d` / ResidualBlock up-down
- Elementwise: add, mul, cat, chunk, clamp, sigmoid, softplus, tanh
- `F.pixel_unshuffle` / `pixel_shuffle` (export as reshape+transpose)
- QP bank: `index_select` → runtime gather outside engine (input binding)

## Must stay on Host

- `compress_prior_2x` / `compress_prior_4x` AR control flow
- DPB update / `prepare_feature_adaptor_i` / `reset_interval`
- NAL SPS / I / P mux-demux (`stream_helper.py`)
- rANS encode/decode + dual-coder merge for `ec_part`
- CDF table load from assets

## Channel / shape constants

| Symbol           | Value                         |
| ---------------- | ----------------------------- |
| `g_ch_src_d`     | 192 (3×8×8)                   |
| `g_ch_recon`     | 320                           |
| `g_ch_y` (inter) | 128                           |
| `g_ch_z`         | 128                           |
| `g_ch_d`         | 256                           |
| Intra `N`        | 256                           |
| QP bank          | 64 (+ `extra_qp` for P-shift) |
| Pad multiple     | 16                            |
| Batch            | 1                             |
