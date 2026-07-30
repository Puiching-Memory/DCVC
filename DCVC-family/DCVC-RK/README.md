# DCVC-RK

A DCVC-UF derivative **redesigned at the model layer for Rockchip RKNN (rk3588)**.
Goal: maximum execution efficiency on the NPU, not maximum weight-transfer
compatibility. Every design choice is backed by a measurement on the actual
toolkit/NPU, and the whole export+build pipeline is automated so it cannot drift
out of sync.

## Design (and the evidence behind each change)

| DCVC-UF construct                                | RKNN behavior (measured)                                         | DCVC-RK replacement                                                          |
| ------------------------------------------------ | ---------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `SubpelConv2x`: Conv→PixelShuffle                | re-lowered to **ConvTranspose** (~29% of `intra_synthesis` time) | **Conv → `F.interpolate` (nearest)** → native NPU `Resize` (~free, 0 cycles) |
| PixelShuffle → `DepthToSpace`                    | RKNN re-lowers it to ConvTranspose anyway                        | use `Resize`, not `DepthToSpace`                                             |
| `WSiLUChunkAdd` FFN (swish + channel-stride add) | many **Slice/Split** memory ops                                  | **GLU FFN** → fused to native `exGlu` op                                     |
| `pixel_unshuffle` downsampling                   | Reshape+Transpose land on **CPU**                                | **stride-2 Conv**, single NPU op                                             |
| `WSiLU = sigmoid(4x)·x`                          | RKNN folds to one `exSwish`                                      | **kept unchanged** (activation weights transfer)                             |
| `dc` branch (pw→dw3x3→pw)                        | NPU-native, no fallback                                          | **kept identical** (weights transfer)                                        |
| `cat([a,b])`→`Conv(2C→C)` adaptor                | a11 TileChannel demotes `ic=2C` (8 weight banks) to single-core  | **split-adaptor** `Conv(C→C)(a)+Conv(C→C)(b)`, same FLOPs → 3-core           |

### GLU FFN with split-expand (the 3-core fix)

GLU replaces UF's `WSiLUChunkAdd`:

```python
class GLUFFN(nn.Module):
    def __init__(self, ch):
        self.ea = nn.Conv2d(ch, ch, 1)   # NOT Conv(ch, ch*2)
        self.eb = nn.Conv2d(ch, ch, 1)
        self.c  = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        return self.c(self.ea(x) * torch.sigmoid(self.eb(x))) + x
```

Why the expand is **two `Conv(C→C)` instead of one `Conv(C→2C)+chunk`**: the RKNN
3-core compiler replicates a layer's activation buffer across 3 cores into SRAM. A
`2C`-wide buffer at f-resolution (136×240) exceeds the SRAM budget and the layer is
demoted to single-core ("3Core fallback"). Two `C`-width convs halve the peak tensor
and clear the threshold. The math is identical (a merged `(2C,C,1,1)` weight splits
row-wise into `ea`/`eb`), so the merged weights transfer 1:1 during finetuning. The
split form also fuses more cleanly to native `ConvSigmoid` + `ConvMul`.

### Channel widths chosen for 3-core (the depthwise fix)

The only other op that demotes is the **depthwise 3×3 inside a DCB**, and **only at
384 channels** (swept: 128/192/256/320 are all 3-core; 384 is not). Two structural
sources of 384 were reduced:

- intra `g_ch_enc_dec`: 384 → **256** (`src/models/image_model.py`)
- `prior_fusion` / `spatial_prior` internal DCB width: 384 → **320**
  (`src/models/video_model.py`); their input/output interfaces stay 384/512/256

These are capacity changes that require training — that is the explicit RK trade-off.

## Automated pipeline (single source of truth)

`build_rknn.py` runs the whole flow with one command and **cannot** bake a stale
shape into a model:

```
model definitions (g_ch_* channel widths)
        │  export_rk.py derives shapes
        ▼
onnx/models_rk/shapes.json   ← single source of truth
        │  build_rknn.py reads, never hardcodes
        ▼
bake concrete dims → load → build → export .rknn
```

This eliminates the bug class where a hand-maintained second shape table (e.g. an
old `384` quant-step width) diverges from the model after a width change, producing
an `Incompatible dimensions` error mid-build. Verified: flipping intra 256↔384 and
rebuilding re-derives the shapes correctly and never crashes.

```bash
# full pipeline: re-export ONNX + build RKNN + analyze + numerics + zip
python DCVC-family/DCVC-RK/build_rknn.py --verify --pack out.zip

# reuse existing ONNX, build only some subnets
python DCVC-family/DCVC-RK/build_rknn.py --no-reexport --subnets recon_generation inter_decoder
```

`build_rknn.py` reports per-subnet: build rc, 3-core-fallback count, internal memory,
CPU ops, and **fallback layer roles** (depthwise3x3 / glu_expand / 1x1_adaptor / head).
RKNN verbose output bypasses Python's stdout (it writes to the raw fd), so the log is
captured via `os.dup2` redirection.

## Verified results (rk3588, 1080p = 1088×1920)

15 subnets (4 intra + 11 inter), GLU + split-expand + 256/320 widths + split-adaptor:

| metric                 | value                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| build success          | **15/15**                                                        |
| `ConvTranspose` ops    | **0**                                                            |
| CPU fallback ops       | **0** (only `InputOperator`/`OutputOperator`, which are pure IO) |
| 3-core fallbacks       | **3** (down from 41 baseline; see below)                         |
| RKNN-sim vs ORT cosine | **1.0000** on all 15 (graph fidelity on random weights)          |

The 3 remaining fallbacks are all **graph-context** demotions of cheap layers (1
depthwise in `intra_synthesis`, 2 `1×1 adaptor`s in `prior_fusion`/`recon`):
isolated, each layer is 3-core; only the full-graph cost-model demotes it. They run
on the NPU regardless (not CPU), and are depthwise/1×1 (minimal cost), so they are
accepted.

### Fallback reduction history

| stage                          | total fallbacks                                    |
| ------------------------------ | -------------------------------------------------- |
| baseline (DCVC-UF ops on RKNN) | 41                                                 |
| + GLU split-expand             | 26 (all inter GLU-expand cleared)                  |
| + 384→256/320 widths           | 3 (on a8 toolkit)                                  |
| toolkit a8→a11                 | 7 (a11 cost-model demotes `ic=512` concat-adaptor) |
| + split-adaptor (cat→split)    | **3** again (= a8; a11 now usable + TC_COST)       |

## Why `single_core_mode=True` is NOT the answer

That config flag removes the fallback **warnings** but by forcing *every* layer to
single-core — including the ~200 layers that legitimately benefit from 3-core
parallelism. It trades real throughput for a clean log. All exposed config knobs
(`optimization_level` 1/2/3, all 4 `memory_plan_strategy` values, `custom_string`,
`op_group_*`) were swept: **none** change the fallback count. The demotion is an
internal compiler cost-model pass with no config-level override, so it is addressed
structurally (split-expand + width) instead.

## Subnets and resolution map

1080p with H padded to a multiple of 64: image `1088×1920`; f = /8 = `136×240`;
y = /16 = `68×120`; z = /64 = `17×30`. Each subnet runs at exactly one of these
resolutions (the ONNX symbols `h`/`h_latent` mean different absolute resolutions per
model — see `shapes.json` for the authoritative per-input shapes).

- **Intra (I-frame):** `intra_synthesis`, `intra_analysis_standard`, `intra_hyper_enc`, `hyper_dec`
- **Inter (P-frame):** `inter_feature_adaptor_i/p`, `inter_feature_extractor`, `inter_encoder`, `inter_hyper_enc`, `inter_hyper_dec`, `inter_temporal_prior`, `inter_prior_fusion`, `inter_spatial_prior`, `inter_decoder`, `recon_generation`

The structure of every subnet mirrors the DCVC-UF checkpoint
(`checkpoints/cvpr2025_video.pth.tar` / `cvpr2025_image.pth.tar`); only the
RKNN-hostile ops change. Note the shipped `src/models/*` Python is **stale** vs the
checkpoint (e.g. wrong `dcb2` flags, `pixel_unshuffle` downsampling) — the checkpoint
weight shapes are the authoritative blueprint.

## Layout

```
DCVC-family/DCVC-RK/         (mirrors the family src/ layout; no __init__.py)
├── src/layers/layers.py     WSiLU, ResizeUpsampleRK, GLUFFN (split-expand),
│                            DepthConvBlockRK, ResidualBlockUpsampleResize
├── src/models/image_model.py 4 intra subnets (widths: y=256, enc_dec=256, z=128)
├── src/models/video_model.py 11 inter subnets (y=128, z=128, d=256, recon=320)
├── export_rk.py             legacy TorchScript exporter (opset 17) → RKNN ONNX
│                            + writes shapes.json (single source of truth)
├── build_rknn.py            automated bake→build→analyze→verify→pack pipeline (fp)
├── build_rknn_i8.py         same pipeline, INT8 (w8a8) with do_quantization=True
├── qat.py                   STE fake-quant (u8 act / s8 weight) for QAT→int8 export
├── profiling/               rknn_hotspot / rknn_tile_probe / adaptor_microbench
├── requirements.txt         runtime deps (torch, onnx, onnxruntime, numpy)
├── checkpoints/             placeholder for the DCVC-UF checkpoint (see HANDOFF)
└── README.md                this file
```

> Build artifacts (`onnx/models_rk/`, `*.rknn` dirs, `shapes.json`) are written to
> the workspace root (`/root/workspace/DCVC`), not inside this folder — the scripts
> locate it automatically, so run them from the workspace root as shown above.

## Next steps

- **RD training** from the DCVC-UF checkpoint: the dc branch + WSiLU + GLU merged
  weights transfer; Resize-upsample, split-expand and width-changed blocks train.
  The RK model is API-compatible with the DCVC-UF training loop
  (`train_image.py` / `train_video.py`).
- **On-board `eval_perf()`** to turn the op-count/MACC wins into measured ms per
  subnet and locate the true wall-time bottleneck (expected: `recon_generation`,
  `inter_decoder`, `inter_encoder` — the large conv stacks).
- **Optional**: sub-graph fusion of the parameter-prediction chain
  (`hyper_decoder + temporal_prior + prior_fusion + spatial_prior`) into one NPU
  call, if profiling shows host↔device transfer dominates the small subnets.
