# DCVC CPU ONNX codec (cross-platform, TensorRT-free I-frame + P-frame codec)

A standalone, **Linux + Windows**, CPU-only end-to-end DCVC-RT I-frame + P-frame
(intra + inter) codec.
It avoids the original TensorRT/CUDA path and the custom-op route by decomposing the
fused `DcvcDepthConv` blocks into standard ONNX operators, then running every
neural network with the **ONNX Runtime CPU execution provider**. Entropy coding
uses the same C rANS library as the native runtime, so the pipeline is fully
independent of TensorRT.

## Directory layout

```
onnx/
├── CMakeLists.txt      # modular build: auto-fetches ONNX Runtime, builds 3 exes, `dcvc_package` target
├── src/                # codec sources (engine, AR codec, intra+inter pipelines, npy reader)
├── tests/              # test executables
├── python/             # model export / standard-op conversion / tolerance sweep
├── scripts/            # build_linux.sh, build_windows.ps1
└── models/             # runtime *.onnx models + *.npy CDF tables (generated)
```

## Why standard ops instead of custom ops?

The exported `intra_analysis.onnx` uses a proprietary `DcvcDepthConv` custom op.
Implementing a CPU custom op for ONNX Runtime adds complexity and a fragile C API
registration path. Since the block is just a composition of `Conv`, `Sigmoid`,
`Mul`, `Add`, and `Slice`, we convert it to standard ONNX operators and let the
CPU EP execute everything in FP32.

## Prerequisites

**Build**
- CMake ≥ 3.18
- A C++17 compiler: GCC/Clang on Linux, or Visual Studio 2019/2022 with the
  "Desktop development with C++" workload on Windows.
- ONNX Runtime is **downloaded automatically** during configure (1.19.0, x64).
  If the direct GitHub download is slow, point the build at a mirror or a
  pre-downloaded archive with these CMake variables:
  - `-DDCVC_ORT_ARCHIVE=/path/to/onnxruntime-*.zip` — use a local file (fastest)
  - `-DDCVC_ORT_URL_BASE=https://gh-proxy.com/` — prefix a GitHub mirror host
  - `-DDCVC_ORT_URL=https://.../onnxruntime-*.zip` — a full custom URL

**Regenerating models** (optional — `models/` ships pre-generated)
- Python 3 with `torch`, `onnx`, `numpy` (run via `uv` or `pip`).

## Build

### Linux
```bash
cd onnx
bash scripts/build_linux.sh
# or, manually:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Windows (x64 Native Tools / PowerShell)
```powershell
cd onnx
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
# or, manually:
cmake -S . -B build -A x64
cmake --build build --config Release
```

Binaries are placed directly under `build/` (Linux) or `build/Release/`
(Windows). The ONNX Runtime shared library is copied next to them automatically,
and on Linux an `$ORIGIN` RPATH is set so they run without `LD_LIBRARY_PATH`.

### Cross-compile Windows from Linux (MinGW-w64)

You can produce Windows x64 binaries without a Windows machine, using MinGW-w64:
```bash
# 1. Install the toolchain (Debian/Ubuntu)
sudo apt-get install mingw-w64

# 2a. Fastest path: pre-download the Windows ORT package via a mirror, then point
#     the build at the local archive (avoids slow direct GitHub downloads):
curl -L -o ort.zip \
  https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-win-x64-1.19.0.zip
DCVC_ORT_ARCHIVE=ort.zip bash onnx/scripts/build_windows_cross.sh

# 2b. Or let the build fetch the Windows ORT package through a mirror:
DCVC_ORT_URL_BASE=https://gh-proxy.com/ bash onnx/scripts/build_windows_cross.sh
```

The MinGW exes statically link the C/C++/pthread runtimes (`-static`), so the
package only needs `onnxruntime.dll` + the models — no MinGW runtime DLLs to
ship. The produced `.exe` files are real PE binaries that run on any Windows x64
machine (and can be verified on the Linux host with `wine ./test_cpu_end2end.exe
../models 256 256 32`).

## Package a runnable folder

The `dcvc_package` target assembles a self-contained `dist/dcvc_onnx_codec/`
folder with the executables, the ONNX Runtime shared lib, all models, and the CDF
tables — ready to copy to any machine of the same OS.

```bash
# Linux
cmake --build build --target dcvc_package
# Windows
cmake --build build --config Release --target dcvc_package
```

Run from inside the packaged folder (models are looked up in the current dir):
```bash
./test_cpu_end2end . 256 256 32          # Linux
test_cpu_end2end.exe . 256 256 32        # Windows
```
`H` and `W` may be **any positive integers** -- the codec pads to the next
multiple of 64 internally and crops back (see *Dynamic resolution* below).

## I-frame + P-frame (inter-frame prediction)

Frame 0 of a sequence is encoded as an **I-frame** by the intra image model
(`DMCI`, `intra_*` models). Frames 1..N-1 are encoded as **P-frames** by the
video model (`DMC`, `inter_*` models), each predicting from the *previous
reconstructed frame* (closed-loop, so encoder and decoder references stay
bit-exact — verified across Linux and Windows exactly like the intra codec).

Pipeline (`src/cpu_inter_pipeline.c`):

```
feature_adaptor_i(pixel_unshuffle(x_hat_prev))      # ref frame -> feature
  -> feature_extractor(feature, q_feature)           # ctx, ctx_t
  -> inter_encoder(pixel_unshuffle(x), ctx, q_enc)   # y latent
  -> hyper_encoder(y) -> z (int8)                    # z bit-estimator (rANS)
  -> [hyper_decoder(z) , temporal_prior(ctx_t)]      # hierarchical + temporal
  -> prior_fusion -> params
  -> 2x checkerboard AR prior (2 rANS passes)         # y symbols
  -> inter_decoder(y_hat, ctx, q_dec)                 # feature
  -> recon_generation(feature, q_recon)               # pixel_shuffle_8 -> x_hat
```

Run a closed-loop I+P sequence (frame 0 intra, the rest inter):

```bash
./test_cpu_inter 5 256 256 32 32      # N H W qp_i qp_p  (any H,W)
```

On real video this gives a large rate saving vs coding every frame as I-frame
(e.g. ~73% fewer bytes at equal/higher PSNR over a 24-frame window).

## Dynamic resolution

The codec accepts **any positive `H` and `W`**; it no longer requires multiples
of 64. Each pipeline pads the input to the next multiple of 64 with edge
replication (`F.pad(mode="replicate")` semantics), runs the network on the
padded frame, then crops the reconstruction back to the original size. The
bitstream stores the original `H,W` so the decoder reconstructs exactly the
requested resolution.

`intra_analysis_standard.onnx` is the only model whose original export baked in a
fixed pixel-unshuffle shape. `python/make_intra_analysis_dynamic.py` rewrites
those two static Reshape constants with a shape-computing subgraph (every Conv
weight is untouched), so the model runs at any spatial size. `export_all_models.py`
runs this patch automatically.

Verified bit-exact round-trips include 64x64, 100x100, 192x256, 200x200,
270x180, 300x200, 320x192 and 512x512 for I-frames, and 128x128 ... 512x512 for
P-frames. On real video at a non-64-multiple size (480x270) the codec encodes
and decodes with no extra loss from the padding crop.

## Run the tests

```bash
cd build                                    # or build\Release on Windows
./test_intra_analysis                        # analysis parity vs golden tensors
./test_ar_cpu                                # AR encode/decode round-trip (default 8x8)
./test_cpu_end2end                           # full I-frame encode/decode round-trip (256x256, qp=32)
./test_cpu_end2end . 200 200 32               # non-64-multiple H,W works too
```

The test binaries auto-detect the model directory (they probe `.`, `../models`,
then `models`), so the model-dir argument is optional -- it is only needed when
the models live elsewhere. `test_cpu_end2end` also accepts real data and a
reference tensor for comparison:
```bash
./test_cpu_end2end 256 256 32 input.npy out.npy ref.npy
```

## Regenerate the models

```bash
cd onnx
uv run python python/export_all_models.py
# Custom locations (defaults are relative to the repo root, never hardcoded):
#   --out-dir <dir>          default: <repo>/onnx/models
#   --checkpoint <pth>       default: <repo>/checkpoints/cvpr2025_image.pth.tar
#   --original-intra-analysis default: <repo>/tensorRT/assets/onnx/intra_analysis.onnx
```

This writes `intra_analysis_standard.onnx` (via `convert_to_standard_ops.py`),
the remaining standard-op networks, the QP scales, and copies the entropy CDF
tables into `models/`.

## Runtime model files (`models/`)

| File | Purpose |
|------|---------|
| `intra_analysis_standard.onnx` | image → latent `y` |
| `intra_hyper_enc.onnx` / `hyper_dec.onnx` | hyper-encoder / decoder |
| `y_prior_fusion.onnx` | fused prior parameters |
| `y_spatial_prior*.onnx` (+ `reduction`, `adaptor_1..3`) | AR spatial prior nets |
| `intra_synthesis.onnx` | latent → reconstruction |
| `gaussian_*.npy` | y rANS CDF tables |
| `bitest_*.npy` | z rANS CDF tables |
| `q_scale_enc.npy` / `q_scale_dec.npy` | per-QP quantization scales |

## Cross-platform interoperability test

`test_cpu_end2end` supports `--encode`/`--decode` modes that persist the codec
bitstream to a self-describing file (`DCV1` magic + H + W + qp + stream), so you
can encode on one OS and decode on another to verify the bitstream is portable.

```bash
# 1. On Windows: encode a frame -> frame.bin (+ frame.bin.enc.npy reconstruction)
test_cpu_end2end.exe --encode frame.bin

# 2. Copy frame.bin (and frame.bin.enc.npy) to Linux

# 3. On Linux: decode it and compare against the Windows reconstruction
./test_cpu_end2end --decode frame.bin frame.bin.enc.npy
#   decode-vs-ref x_hat max_diff=0.000000  -> bitstream is fully portable
```

The reverse direction (encode Linux, decode Windows) works identically. In
practice the bitstreams and reconstructions come out **byte-identical** across
Linux and Windows for the same input.

### Why is it bit-exact, given the prior tolerance is only ~1e-6?

Two different quantities are at play: (a) the prior-mismatch *tolerance* (how
much synthetic noise breaks the encoder/decoder, measured at ~1e-6 by
`python/sweep_prior_tolerance.py`), and (b) the *actual* Windows-vs-Linux ONNX
Runtime FP difference. These are not the same. The decode path reconstructs the
prior locally (`hyper_dec` + `prior_fusion`) -- the prior is **not** in the
bitstream -- so byte-identical output implies the prior itself is bit-identical.

Measured directly (GCC-built Linux ORT vs MSVC-built Windows ORT, same models,
~197k floats dumped from both `y` and `params_fusion`):

| tensor | elements | byte-identical | max abs diff |
|--------|----------|----------------|--------------|
| `y` (analysis out) | 65,536 | yes | 0.000e+00 |
| `params_fusion` (prior) | 131,584 | yes | 0.000e+00 |

i.e. the actual cross-ORT difference is **0**, ~1,000,000x under the 1e-6
divergence threshold, so there is nothing to amplify. The ONNX Runtime CPU EP
is deterministic for these small, fixed-shape conv/gemm kernels (no
non-deterministic reductions), and the Linux `.so` and Windows `.dll` builds
run the same arithmetic on x86-64. This is an empirical property of this
model/version/ISA, not a universal guarantee.

You can reproduce the measurement on your own two machines:
```bash
# encode side (dump the prior to compare across OSes)
DCVC_DUMP_Y=y.npy DCVC_DUMP_PARAMS=params_fusion.npy \
    ./test_cpu_end2end --model-dir <dir> --encode frame.bin
# then compare y.npy / params_fusion.npy element-wise between the two OSes
```

## Accuracy

- The C++ encode/decode loop is bit-exact: encoder and decoder reconstructions
  are identical (`max_diff = 0`).
- Versus the PyTorch FP32 reference, the prior networks carry a small ONNX
  Runtime numerical difference (per-prior max ~5e-5), which propagates to a small
  reconstruction difference (avg ~1.5e-4 on [0,1] images). This is a tolerance
  difference, not a correctness issue: encoder and decoder share the same models.

## Model details

`DepthConvBlock` expands to `dc3(ws_relu(dc2(ws_relu(dc0(x))))) + x` followed by
`ffn2(ws_relu_chunk_add(ffn0(dc)))`, where
`ws_relu(x) = x * sigmoid(4*x)` and `ws_relu_chunk_add(x) = ws_relu(x[:,:2C]) + ws_relu(x[:,2C:])`.
`convert_to_standard_ops.py` replaces every `DcvcDepthConv` node with this exact
subgraph using ONNX `Conv`, `Sigmoid`, `Mul`, `Add`, and `Slice`, so the result
runs on the CPU EP with no custom code.
