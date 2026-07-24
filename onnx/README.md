# DCVC CPU ONNX codec (cross-platform, TensorRT-free I-frame + P-frame codec)

A standalone, **Linux + Windows** end-to-end DCVC-RT I-frame + P-frame
(intra + inter) codec. CPU by default, with optional CUDA/TensorRT acceleration.
Every neural subnet is exported from PyTorch as standard ONNX operators (no
custom CUDA ops) and runs under **ONNX Runtime** (CPU execution provider by
default; the CUDA or TensorRT EP can be enabled at runtime, see "GPU execution"
below). Entropy coding uses the same C rANS library as the native runtime, so the
pipeline is fully independent of TensorRT.

## Directory layout

```
onnx/
├── CMakeLists.txt      # modular build: auto-fetches ONNX Runtime, builds 3 exes, `dcvc_package` target
├── src/                # codec sources (engine, AR codec, intra+inter pipelines, npy reader)
├── tests/              # test executables
├── python/             # model export / PTQ / tolerance sweep
├── scripts/            # build_linux.sh, build_windows.ps1
└── models/             # runtime *.onnx models + *.npy CDF tables (generated)
```

## Why standard ops instead of custom ops?

The TensorRT path fuses each `DepthConvBlock` into a proprietary `DcvcDepthConv`
custom op. The CPU pipeline instead exports every subnet (including
`IntraEncoder`) via torch dynamo from the pure-PyTorch `forward_torch` path —
`Conv` / `Sigmoid` / `Mul` / `Add` / `Split` — so the ORT CPU EP runs everything
in FP32 with no custom-op registration.

## Prerequisites

**Build**
- CMake ≥ 3.18
- A C++17 compiler: GCC/Clang on Linux, or Visual Studio 2019/2022 with the
  "Desktop development with C++" workload on Windows.
- ONNX Runtime is **downloaded automatically** during configure (1.27.0, x64).
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

### GPU execution (optional)

By default everything runs on the ONNX Runtime CPU execution provider. To
enable the CUDA or TensorRT execution provider:

```bash
# 1. Build against the GPU-enabled ONNX Runtime package (fetches
#    onnxruntime-*-gpu_cuda12-1.27.0 instead of the CPU one). The system must
#    have matching CUDA + cuDNN installed (plus TensorRT for the TensorRT EP);
#    see the ONNX Runtime 1.27 release notes for exact versions.
#    For CUDA 13, add -DDCVC_ORT_CUDA_VERSION=13.
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DDCVC_ORT_GPU=ON
cmake --build build-gpu -j$(nproc)

# 2. Select the execution provider at runtime:
DCVC_USE_GPU=1 ./build-gpu/test_cpu_end2end   # CUDA EP
DCVC_USE_GPU=2 ./build-gpu/test_cpu_end2end   # TensorRT EP
DCVC_USE_GPU=0 ./build-gpu/test_cpu_end2end   # CPU (default)
DCVC_GPU_DEVICE=1 DCVC_USE_GPU=1 ./build-gpu/test_cpu_end2end  # second GPU
```

If the requested provider is not compiled into the fetched ONNX Runtime (e.g.
`DCVC_USE_GPU=1` with the default CPU package) the engine prints a warning and
falls back to CPU. C API users can pass the same `0/1/2` values as the
`use_gpu` argument of `dcvc_cpu_engine_create()`, which takes precedence over
`DCVC_USE_GPU`.

Note: entropy coding (rANS) and all tensor buffers stay on the CPU; ONNX
Runtime inserts the device copies automatically. The codec runs many small
sessions per frame, so PCIe transfer overhead can eat the GPU speedup —
measure against the CPU build before committing to it.

Cross-device caution: mixing execution providers between encoder and decoder
(e.g. CPU encode, GPU decode) can desync the rANS entropy coder via ULP-level
FP differences in the entropy-parameter networks when those nets are FP32.
The default path uses **fixed-point** `com.dcvc` ops so CPU encode/decode stays
cross-OS bit-exact — see
[docs/fxp_cross_platform_rans.md](docs/fxp_cross_platform_rans.md).
Background and failed INT8/INT16 PTQ attempts:
[docs/entropy_sync_ptq_report.md](docs/entropy_sync_ptq_report.md).

Fixed-point entropy nets (ORT custom ops) — now the DEFAULT in `onnx/models/`.
Intra + inter entropy nets (and key inter recon/feature nets) use
`com.dcvc::FxpConv` + `com.dcvc::FxpWsRelu` (int16 act / int16 weights, LUT
nonlinear). Integer arithmetic is associative, so the entropy path is
cross-platform bit-exact by construction. CDF index selection uses a shared
integer rule (`src/fxp/fxp_scale_index.c`): float→Q18→Q12→128-level scale-table
lookup — no runtime `logf`/`floorf`. Encode and decode both call this path.
(The FP32 nets are in `models_fp32/` for RD comparison only.)
```bash
# Regenerate the fxp nets into a directory (real-content activation calibration):
uv run python python/fxp_export_entropy_nets.py --calib-dir <calib> --out-dir models_fxp
cmake --build build --target test_fxp_scale_index test_cpu_end2end
./build/test_fxp_scale_index
./build/test_cpu_end2end models_fxp 128 128 32
```

### Cross-compile Windows from Linux (MinGW-w64)

You can produce Windows x64 binaries without a Windows machine, using MinGW-w64:
```bash
# 1. Install the toolchain (Debian/Ubuntu)
sudo apt-get install mingw-w64

# 2a. Fastest path: pre-download the Windows ORT package via a mirror, then point
#     the build at the local archive (avoids slow direct GitHub downloads):
curl -L -o ort.zip \
  https://gh-proxy.com/https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-win-x64-1.27.0.zip
DCVC_ORT_ARCHIVE=ort.zip bash onnx/scripts/build_windows_cross.sh

# 2b. Or let the build fetch the Windows ORT package through a mirror:
DCVC_ORT_URL_BASE=https://gh-proxy.com/ bash onnx/scripts/build_windows_cross.sh
```

The MinGW exes statically link the C/C++/pthread runtimes (`-static`), so the
package only needs `onnxruntime.dll` + the models — no MinGW runtime DLLs to
ship. The produced `.exe` files are real PE binaries that run on any Windows x64
machine. To assemble a ready-to-distribute Windows folder from the cross build:

```bash
cmake --build onnx/build-mingw --target dcvc_package
# -> onnx/dist/dcvc_onnx_codec/*.exe + onnxruntime.dll + models
```

Zip `onnx/dist/dcvc_onnx_codec/` and ship it — no compiler is needed on the
Windows side.

> **⚠️ Wine is a smoke test, not a cross-platform proof.** A quick sanity check
> on the Linux host with `wine ./test_cpu_end2end.exe ../models 256 256 32` (and
> even a Wine cross-decode round-trip) is useful — it confirms the PE links
> correctly, `onnxruntime.dll` loads, and the logic is self-consistent. But Wine
> is a *Windows API compatibility layer, not a CPU emulator*: the `.exe`'s
> x86-64 instructions execute natively on the same Linux CPU, so it does **not**
> exercise the real cross-platform risk surface — a different OS, a different
> compiler (the production Windows path is MSVC via `build_windows.ps1`, not
> MinGW), a different CPU microarchitecture, or a different ORT build. A Wine
> round-trip of `max_diff=0` only says "the binary is not broken"; it says
> nothing about whether MSVC-built Windows binaries will match GCC-built Linux
> binaries. For a genuine cross-platform guarantee, run the `--encode`/`--decode`
> interop test below on two *physical* machines with different OSes (see
> *Cross-platform interoperability test*).

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

### Double-click on Windows (console stays open)

The Windows test `.exe`s detect when they were launched by **double-click**
(parent process is `explorer.exe`) and pause with `[Press Enter to exit]` before
returning, so the console window does not vanish and the output can be
read/copied. When run from a terminal, a script, or with redirected stdin/pipe,
they return immediately and never block automation.

- Set `DCVC_FORCE_PAUSE=1` to force the pause even from a terminal (handy for
  capturing output). Redirection/pipes are always skipped to avoid hanging.
- This logic (`onnx/tests/console_pause.h`) is a no-op on Linux/macOS.


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

All exported models use dynamo `dynamic_shapes` so spatial dims are symbolic;
`intra_analysis_standard.onnx` accepts any `H,W` (multiple of 8 for the
pixel-unshuffle, and the full pipeline pads to a multiple of 64).

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
```

This dynamo-exports all 9 intra nets from `DMCI` (including
`intra_analysis_standard.onnx` from `model.enc`), writes the QP scales, and
copies the entropy CDF tables into `models/`.

## Runtime model files (`models/`)

`onnx/models/` ships the **fixed-point (fxp) entropy nets by default**: the 7
entropy-parameter networks use integer arithmetic (`com.dcvc` custom ops), which
is cross-platform bit-exact by construction. This is the path real cross-platform
testing is done against. The original FP32 entropy nets are kept in
`onnx/models_fp32/` for RD comparison only — they can desync across MSVC/GCC on
real video (see `docs/entropy_sync_ptq_report.md`).

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

> This is the **real** cross-platform test — unlike the Wine smoke check above,
> it requires two *physical* machines of different OSes (Linux and Windows) and
> is the only way to validate the actual GCC-vs-MSVC and Linux-vs-Windows
> floating-point differences. Run it with MSVC-built Windows binaries
> (`build_windows.ps1`, not the MinGW cross build) for a production-faithful
> result. The `max_diff=0` measurement in the table below came from this path
> (GCC Linux ORT vs MSVC Windows ORT), not from Wine.

Because `onnx/models/` now ships the **fixed-point entropy nets by default**,
the entropy path is integer-arithmetic (bit-exact by construction), which is
what makes cross-platform encode/decode reliable. (With the FP32 nets in
`models_fp32/`, real cross-platform P-frame sequences can desync — confirmed by
testing: an MSVC-encoded 10-frame P sequence fails to decode on GCC at frame 3
with `entropy_sync`.)

`test_cpu_end2end` (I-frame) and `test_cpu_inter` (I+P sequence) both support
`--encode`/`--decode` with self-describing containers (`DCV1` for a single
frame, `DCVS` for a sequence), so you can encode on one OS and decode on
another to verify the bitstream is portable.

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

`DepthConvBlock.forward_torch` expands to
`dc3(ws_relu(dc2(ws_relu(dc0(x))))) + x` followed by
`ffn2(ws_relu_chunk_add(ffn0(dc)))`, where
`ws_relu(x) = x * sigmoid(4*x)` and
`ws_relu_chunk_add(x) = ws_relu(x[:,:2C]) + ws_relu(x[:,2C:])`. Torch dynamo
exports that subgraph as ONNX `Conv` / `Sigmoid` / `Mul` / `Add` / `Split`, so
the result runs on the CPU EP with no custom code.
