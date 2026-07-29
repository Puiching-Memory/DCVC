# DCVC-SDK Quick Start

`libdcvc` is the cross-platform, bit-exact neural video codec built on ONNX
Runtime + fixed-point custom ops (`com.dcvc`) + rANS entropy coding. It exposes
a small stable C ABI and ships as a versioned shared library.

## What you get

```
include/dcvc/*.h          public headers
lib/libdcvc.so.1.0.0      shared library (+ .so.1 / .so SONAME symlinks)
lib/cmake/dcvc/           find_package(dcvc) support
models/<name>/            discovered/configured resolution model packs
```

Ship `libdcvc.so.*` and `libonnxruntime.so.*` next to your application binary
(the library uses an `$ORIGIN` runtime path, so co-locating them is enough).

## Build the SDK

```bash
cmake -S onnx -B onnx/out/build/linux-x64 \
  -DCMAKE_BUILD_TYPE=Release -DDCVC_FXP_CUDA=OFF
cmake --build onnx/out/build/linux-x64 --target dcvc
cmake --install onnx/out/build/linux-x64  # -> /usr/local by default
```

Options: `-DDCVC_FXP_CUDA=ON` (CUDA FXP backends), `-DDCVC_ORT_GPU=ON`
(CUDA/TensorRT execution providers), `-DDCVC_ORT_CUDA_VERSION=12`.

The portable package scripts discover and bundle every `models_<resolution>`
directory by default. Thus the current 720p and 1080p packs are included, and
a future `models_2160p` is included without changing the scripts:

```bash
bash onnx/scripts/package.sh
# Windows PowerShell: onnx\scripts\package.ps1
```

Generated files are separated by purpose:

```text
onnx/out/build/<platform>/       CMake intermediates
onnx/out/staging/<package>/      temporary; removed after successful archive
onnx/out/runnable/<platform>/    unpacked runnable codec
onnx/out/packages/               final versioned archives
```

Use repeatable `--model-pack name=directory` arguments (PowerShell:
`-ModelPack name=directory`) to replace discovery with an explicit pack set,
or `--no-models` / `-NoModels` for a library-only package.

## Integrate (CMake)

```cmake
find_package(dcvc 1.0 CONFIG REQUIRED)
target_link_libraries(myapp PRIVATE dcvc::dcvc)
```

Headers are wired into the target; no extra `include_directories` is needed.

## Integrate (plain compiler)

```bash
cc app.c -o app -I<prefix>/include -L<prefix>/lib -ldcvc
```

## Encode / decode

```c
#include <dcvc/dcvc.h>

/* session: owns the model directory, EP selection and custom-op registration */
dcvc_config_t* cfg = dcvc_config_create();
dcvc_config_set_model_dir(cfg, "/path/to/models");
dcvc_session_t* sess = NULL;
dcvc_session_create(cfg, &sess);
dcvc_config_destroy(cfg);

/* encoder: fixed size + base qp; first frame is intra, then P-frames.
 * The DPB (previous reconstruction) is owned by the encoder.
 * width and height must each be a multiple of 64 (e.g. 1920x1088). */
dcvc_encoder_t* enc = NULL;
dcvc_encoder_create(sess, 1920, 1088, 32, &enc);

uint8_t* pkt = NULL; size_t n = 0; dcvc_frame_type_t type;
dcvc_encoder_encode(enc, frame_rgb_fp32, /*force_intra=*/0, &pkt, &n, &type);
/* `pkt` is a self-describing packet: a decoder needs no out-of-band info. */

/* rate control: override the qp per frame (one-time pipeline reload on change) */
dcvc_encoder_encode_qp(enc, frame, 0, 22, &pkt, &n, &type);

/* decoder: dimensions/qp are read from each packet header */
dcvc_decoder_t* dec = NULL;
dcvc_decoder_create(sess, &dec);
int w, h;
dcvc_packet_probe(pkt, n, &w, &h, NULL, &type);     /* allocate out buffer */
float* rec = malloc((size_t)3 * w * h * sizeof(float));
dcvc_decoder_decode(dec, pkt, n, rec, &w, &h, &type);

dcvc_packet_free(pkt);
dcvc_encoder_destroy(enc);
dcvc_decoder_destroy(dec);
dcvc_session_destroy(sess);
```

## Cross-platform bit-exactness

The codec is deterministic across OS / CPU / GPU by construction:

- **FXP fixed-point ops** (`com.dcvc`): integer MAC inside every entropy and
  inter net — integer reduction is associative, so the result does not depend
  on thread count, SIMD width or EP.
- **Integer scale → CDF index** (`fxp_scale_index`): replaces floating
  `log`/`floor` with a fixed Q18→Q12→table rule shared by encoder and decoder.
- **rANS guards**: bounded scan + `bytes_consumed` + optional CRC-32 detect any
  residual desync as `DCVC_ERR_ENTROPY` instead of silent corruption.

A packet produced on Windows decodes bit-identically on Linux (and vice versa).
See `docs/fxp_cross_platform_rans.md` for the full design.

## Packet format (DCV2)

Self-describing, little-endian, identical across hosts:

| offset  | field                         | type    |
| ------- | ----------------------------- | ------- |
| 0       | magic `"DCV2"`                | 4 bytes |
| 4       | version (=2)                  | u8      |
| 5       | flags (bit0 = CRC present)    | u8      |
| 6       | frame_type (0=intra, 1=inter) | u8      |
| 7       | qp                            | u8      |
| 8       | width                         | u32 LE  |
| 12      | height                        | u32 LE  |
| 16      | payload_len                   | u32 LE  |
| 20      | payload                       | bytes   |
| trailer | crc32 (if flags bit0)         | u32 LE  |

## Versioning

`libdcvc.so.MAJOR` — ABI is compatible within a major version. Bump
`DCVC_VERSION_MAJOR` in `include/dcvc/dcvc_version.h` (and the matching
`set(DCVC_VERSION_MAJOR ...)` in `CMakeLists.txt`) on any ABI break.
