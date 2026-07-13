# DCVC-RT Native Runtime

Pure-C API high-performance inference engine for DCVC-RT, backed by TensorRT
(optional) and the existing rANS C++ entropy codec.

## Layout

```
native/
  include/dcvc_rt.h     Public C API
  src/                  Bitstream, color, DPB, TRT runner, orchestrator
  rans/                 rANS (from src/cpp) + C wrapper
  plugins/              Fused op CPU refs + TRT Plugin scaffolding
  tools/                Golden/CDF/weight/engine offline Python tools
  tests/                Unit + container roundtrip tests
  docs/ops_inventory.md Operator → TRT/Plugin map
  assets/               Engines, CDF, QP banks (generated)
```

## Build (CPU container + entropy, no TensorRT)

```bash
cd native
cmake -S . -B build -DDCVC_RT_BUILD_TESTS=ON
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

### With TensorRT

```bash
cmake -S . -B build -DDCVC_RT_HAS_TENSORRT=ON -DTensorRT_ROOT=/path/to/TensorRT
```

## Offline asset pipeline

```bash
# Requires PyTorch + checkpoints under ./checkpoints/
python native/tools/export_cdf.py
python native/tools/convert_weights.py
python native/tools/build_engines.py
python native/tools/dump_golden.py   # needs CUDA
```

## C API sketch

```c
DcvcRtConfig cfg;
dcvc_rt_config_init(&cfg);
cfg.width = 1920; cfg.height = 1080;
cfg.asset_dir = "native/assets";

DcvcRtEncoder* enc = dcvc_rt_encoder_create(&cfg, NULL);
DcvcRtPacket pkt;
dcvc_rt_encode_frame(enc, &frame, &pkt);
```

Without `.engine` files under `assets/engines/`, encode/decode still produce a
valid SPS/I/P container (for integration). Full NN fidelity requires engines
built by `build_engines.py` and Plugin registration.
