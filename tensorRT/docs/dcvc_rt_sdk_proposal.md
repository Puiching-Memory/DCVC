# DCVC-RT Native 推理引擎方案

> 基于 TensorRT + CUDA 的纯 C API 高性能运行时，面向 NVIDIA 设备。
>
> **更新说明**：本文档已根据 `tensorRT/` 目录的实际落地进展更新。初始方案曾评估 ONNX Runtime + TensorRT EP，实际实现采用更贴近硬件的 TensorRT C API + 自研 CUDA Plugin/Kernel 方案。

---

## 1. 设计目标

- 为 DCVC-RT 提供独立的 C 语言推理 SDK，脱离 Python / PyTorch 运行环境。
- 支持 NVIDIA GPU 上的实时 / 近实时视频编码与解码。
- 保留 DCVC-RT 的宽码率、率控、YUV/RGB 统一等核心能力。
- 利用成熟推理引擎降低开发周期与维护成本。

---

## 2. 技术选型

| 层级         | 选型                                                                 | 原因                                                     |
| ------------ | -------------------------------------------------------------------- | -------------------------------------------------------- |
| 基础运行时   | TensorRT C API / ONNX 中间格式                                       | 自动图优化、CUDA Graph、NVIDIA 最优性能                  |
| NVIDIA 加速  | TensorRT Engine + CUDA Kernel                                        | 关键路径 GPU 化，支持动态高宽                            |
| 自定义算子   | TensorRT Plugin + 运行时 `dlopen` CUDA Kernel (`libdcvc_kernels.so`) | 深度卷积、亚像素卷积、pixel_shuffle、mask 处理等融合算子 |
| 熵编码       | 纯 C / C++ rANS（复用 DCVC `src/cpp/py_rans`）                       | 脱离 Python / pybind11，保持 bitstream 兼容              |
| 数据流与状态 | 自研 C                                                               | 帧间状态、DPB、GOP、量化参数、时序上下文                 |
| 对外接口     | 纯 C API (`include/dcvc_rt.h`)                                       | 最大可移植性，便于集成到 C/C++/Go/Rust 等                |

---

## 3. 整体架构

```
┌─────────────────────────────────────────┐
│         dcvc_rt SDK (C API)             │
│  dcvc_rt_encoder_create / encode_frame   │
│  dcvc_rt_decoder_create / decode_packet  │
├─────────────────────────────────────────┤
│      Pipeline (dcvc_intra_pipeline.c     │
│      / dcvc_inter_pipeline.c)             │
│  GOP 管理、DPB、帧间状态、QP bank、时序上下文 │
├─────────────────────────────────────────┤
│      Entropy Codec (rANS C/C++)          │
│  PMF→CDF、算术编码、bitstream 读写         │
├─────────────────────────────────────────┤
│      TensorRT Runner (trt_runner.c)      │
│  引擎加载、输入 shape 绑定、enqueueV3      │
│  Plugin / CUDA Kernel 注册                │
├─────────────────────────────────────────┤
│      CUDA / cuDNN / TensorRT             │
└─────────────────────────────────────────┘
```

---

## 4. 项目目录结构

实际代码位于 `tensorRT/`：

```
tensorRT/
├── CMakeLists.txt
├── README.md
├── include/
│   └── dcvc_rt.h              # 公共 C API
├── src/
│   ├── dcvc_rt.c              # 引擎封装、生命周期
│   ├── trt_runner.c           # TensorRT 会话调度
│   ├── trt_engine.cpp         # TensorRT 引擎加载
│   ├── dcvc_intra_pipeline.c
│   ├── dcvc_inter_pipeline.c
│   ├── dcvc_ar_codec.c
│   ├── bitstream.c
│   ├── color.c
│   └── dpb.c
├── rans/
│   ├── rans.cpp
│   ├── rans_c.cpp             # C API 封装
│   └── pmf_cdf.cpp
├── plugins/
│   ├── include/
│   └── src/
│       ├── plugins_cpu.c      # CPU 参考实现
│       ├── ar_prior.c
│       └── dcvc_kernels.cu    # CUDA 融合核
├── tools/
│   ├── export_cdf.py
│   ├── convert_weights.py
│   ├── build_engines.py
│   └── dump_golden.py
├── tests/
│   ├── test_bitstream.c
│   ├── test_rans.c
│   ├── test_color.c
│   ├── test_roundtrip.c
│   ├── test_intra_pipeline.c
│   ├── test_inter_pipeline.c
│   ├── test_closed_loop.c
│   ├── test_4x_pipeline.c
│   ├── test_trt_engine.c
│   ├── test_sdk_inter.c
│   ├── test_perf.c
│   └── test_quality.c
├── docs/
│   └── ops_inventory.md
└── assets/
    ├── engines/
    ├── cdf/
    ├── qp/
    └── golden/
```

---

## 5. 关键算子与 TensorRT / Plugin 映射

| 算子                                                               | 处理方式                                                      | 说明                                |
| ------------------------------------------------------------------ | ------------------------------------------------------------- | ----------------------------------- |
| Conv2d / DepthwiseConv2d / PixelShuffle / PixelUnshuffle           | TensorRT + 自定义 `DcvcDepthConv` / `DcvcSubpelConv2x` Plugin | 标准算子直接解析，融合算子走 Plugin |
| WSiLU = x * sigmoid(4x)                                            | 折叠到 DepthConv Plugin                                       | 减少节点数                          |
| SubpelConv2x                                                       | `DcvcSubpelConv2x` Plugin                                     | 亚像素运动补偿                      |
| process_with_mask                                                  | `DcvcProcessWithMask` Plugin                                  | 先验与隐变量 mask 处理              |
| bias_pixel_shuffle_8                                               | `DcvcBiasPixelShuffle8` Plugin                                | 解码端 pixel_shuffle 加 bias        |
| build_index_enc / dec                                              | `DcvcBuildIndex` Plugin                                       | rANS 索引构建                       |
| combine_for_reading_2x / restore_y_2x/4x                           | `DcvcCombineRead2x` / `DcvcRestoreY` Plugin                   | AR 读取路径                         |
| round_and_to_int8 / clamp_reciprocal_with_quant / add_and_multiply | `DcvcRoundInt8` / `DcvcClampRecipQuant` / `DcvcAddMul`        | 量化辅助核                          |
| AR 控制流 / DPB / QP gather                                        | Host C 运行时                                                 | 循环、状态管理、NAL mux             |
| rANS encode / decode                                               | CPU rANS（GPU 化已在探索）                                    | 熵编码                              |

完整映射见 [tensorRT/docs/ops_inventory.md](tensorRT/docs/ops_inventory.md)。

---

## 6. C API 设计（实际接口）

```c
#ifndef DCVC_RT_H
#define DCVC_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DcvcRtStatus {
    DCVC_RT_OK = 0,
    DCVC_RT_ERR_INVALID_ARG = 1,
    DCVC_RT_ERR_IO = 2,
    DCVC_RT_ERR_NO_ENGINE = 3,
    DCVC_RT_ERR_CUDA = 4,
    DCVC_RT_ERR_TRT = 5,
    DCVC_RT_ERR_ENTROPY = 6,
    DCVC_RT_ERR_BITSTREAM = 7,
    DCVC_RT_ERR_OOM = 8,
    DCVC_RT_ERR_UNSUPPORTED = 9,
    DCVC_RT_ERR_INTERNAL = 10
} DcvcRtStatus;

typedef enum DcvcRtPixelFormat {
    DCVC_RT_FMT_YUV420P = 0,
    DCVC_RT_FMT_YUV444P = 1,
    DCVC_RT_FMT_RGB24 = 2
} DcvcRtPixelFormat;

typedef enum DcvcRtFrameType {
    DCVC_RT_FRAME_I = 0,
    DCVC_RT_FRAME_P = 1
} DcvcRtFrameType;

typedef struct DcvcRtConfig {
    int width;              /* original width */
    int height;             /* original height */
    int qp;                 /* 0..63 */
    int reset_interval;     /* feature refresh period; default 64 */
    int device_id;          /* CUDA device */
    int use_cuda_graph;     /* 0/1 */
    const char* asset_dir;  /* engines, CDF, weights manifest */
    DcvcRtPixelFormat format;
} DcvcRtConfig;

typedef struct DcvcRtFrame {
    int width;
    int height;
    DcvcRtPixelFormat format;
    /* YUV420: y, u, v planes; YUV444/RGB: data[0] contiguous CHW or HWC per format docs */
    const uint8_t* data[3];
    int stride[3];
    /* Optional FP16 planar YCbCr444 [0,1] already prepared (overrides data[]) */
    const void* ycbcr444_fp16; /* size = 3 * padH * padW * 2, may be NULL */
} DcvcRtFrame;

typedef struct DcvcRtPacket {
    uint8_t* data;
    size_t size;
    DcvcRtFrameType frame_type;
    int qp;
    int sps_written; /* 1 if this packet begins with SPS */
} DcvcRtPacket;

typedef struct DcvcRtEncoder DcvcRtEncoder;
typedef struct DcvcRtDecoder DcvcRtDecoder;

const char* dcvc_rt_status_string(DcvcRtStatus st);
const char* dcvc_rt_version(void);

DcvcRtStatus dcvc_rt_config_init(DcvcRtConfig* cfg);

DcvcRtEncoder* dcvc_rt_encoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st);
DcvcRtStatus dcvc_rt_encode_frame(DcvcRtEncoder* enc,
                                  const DcvcRtFrame* in,
                                  DcvcRtPacket* out);
DcvcRtStatus dcvc_rt_encoder_flush(DcvcRtEncoder* enc, DcvcRtPacket* out);
void dcvc_rt_encoder_destroy(DcvcRtEncoder* enc);

DcvcRtDecoder* dcvc_rt_decoder_create(const DcvcRtConfig* cfg, DcvcRtStatus* out_st);
DcvcRtStatus dcvc_rt_decode_packet(DcvcRtDecoder* dec,
                                   const uint8_t* data,
                                   size_t size,
                                   DcvcRtFrame* out);
void dcvc_rt_decoder_destroy(DcvcRtDecoder* dec);

void dcvc_rt_packet_free(DcvcRtPacket* pkt);
void dcvc_rt_frame_free_planes(DcvcRtFrame* frame);

#ifdef __cplusplus
}
#endif

#endif /* DCVC_RT_H */
```

完整头文件见 [tensorRT/include/dcvc_rt.h](tensorRT/include/dcvc_rt.h)。

---

## 7. 开发阶段（更新至 2026-07-20）

### Phase 0：总体设计与算子清单（已完成）

- 确定 TensorRT + CUDA Kernel 的技术路线。
- 输出 [tensorRT/docs/ops_inventory.md](tensorRT/docs/ops_inventory.md)，明确每个算子去向（TRT / Plugin / Host / CPU）。
- 完成 CMake 构建与测试框架。

### Phase 1：TensorRT 引擎与 Plugin 框架（已完成）

- 离线工具链：`export_cdf.py`、`convert_weights.py`、`build_engines.py`。
- 实现 `DcvcDepthConv`、`DcvcSubpelConv2x`、`DcvcProcessWithMask`、`DcvcBiasPixelShuffle8`、`DcvcBuildIndex` 等 Plugin/Kernel。
- 与 PyTorch 参考实现进行逐算子 parity 验证。

### Phase 2：熵编码与 Bitstream（已完成）

- 将 `src/cpp/py_rans` 改造为 C++ 核心 + C 封装（`rans/`）。
- 实现 `bitstream.c` 与 NAL SPS / I / P 容器，保持与官方 Python 版本 round-trip 兼容。
- `test_roundtrip.c`、`test_ar_codec.c` 通过。

### Phase 3：I 帧 / P 帧 Pipeline 闭环（已完成）

- 完成 Intra (DMCI) 与 Inter (DMC) 全链路编码/解码。
- 实现 DPB 管理、QP bank、4×/2× AR 控制流。
- `test_closed_loop.c`、`test_intra_pipeline.c`、`test_inter_pipeline.c`、`test_sdk_inter.c` 通过。

### Phase 4：GPU 化与性能优化（进行中）

- 引入专用 CUDA 流，重叠 GPU 计算与熵编码。
- 缓存 TRT 引擎输入 shape 绑定，减少 `setInputShape` 开销。
- 将 YUV420 / FP16 色彩转换迁移到 GPU CUDA 核。
- 预加载 QP bank，重叠 rANS 与 GPU 计算。
- 评估 GPU rANS 可行性并记录决策。

### Phase 5：性能基线与交付（待完成）

- 在 NVIDIA GPU 上完成 1080p / 4K 编解码性能基线测试。
- 容器 round-trip 测试与 bitstream 兼容性验证。
- 提供 `examples/` 与 SDK 使用文档。

---

## 8. 方案对比

### 8.1 与全自研 CUDA 内核方案对比

| 维度        | 全自研 CUDA 内核 | TensorRT + 自定义 Plugin/Kernel |
| ----------- | ---------------- | ------------------------------- |
| 开发周期    | 6-9 个月         | 2-3 个月                        |
| NVIDIA 性能 | 取决于手写内核   | 接近最优（TensorRT 自动优化）   |
| 跨平台      | 最好             | 较好（仅依赖 TensorRT/CUDA）    |
| 维护成本    | 高               | 中                              |
| 自定义空间  | 最大             | 中（关键算子可插件化）          |
| 二进制大小  | 最小             | 较大（TensorRT 库）             |

### 8.2 与初始 ONNX Runtime 方案对比

- 原方案使用 ONNX Runtime C API 加载 `.onnx`，通过 TRT EP 加速。
- 实际实现改为 TensorRT C API 直接加载 `.engine`（由 `.onnx` 离线构建），减少一层运行时抽象，便于精细控制 shape 绑定、CUDA 流和 Plugin 注册。
- 保留 ONNX 作为中间表示，模型转换仍通过 Python 工具链完成。

---

## 9. 风险与缓解

| 风险                               | 缓解措施                                                                |
| ---------------------------------- | ----------------------------------------------------------------------- |
| 自定义 Plugin 与 TensorRT 版本绑定 | 使用 CMake 选项 `DCVC_RT_HAS_TENSORRT` 隔离，CI 锁定 TensorRT/CUDA 版本 |
| TensorRT 对动态 shape 支持有限     | 固定 batch=1，高宽按 16 倍数对齐，运行时缓存已构建的 shape 绑定         |
| 自定义算子与 TensorRT 不兼容       | 同时提供 CPU 参考实现（`plugins_cpu.c`）与 CUDA Kernel 兜底             |
| GPU rANS 复杂度与收益不确定        | 先记录决策依据，保留 CPU rANS + 与 GPU 计算重叠的方案                   |
| INT16 确定性推理                   | 作为二期目标，先保证 FP16/FP32 功能正确                                 |
| 熵编码 bitstream 兼容              | 与官方 Python 版本进行 round-trip 对比                                  |

---

## 10. 下一步行动

1. 完成 GPU rANS 可行性验证，或确定 CPU rANS 与 GPU 计算重叠的优化方案。
2. 在 NVIDIA GPU 上完成 1080p / 4K 编解码性能基线测试。
3. 运行容器 round-trip 测试，验证与官方 Python 版本的 bitstream 兼容性。
4. 补充 `examples/` 与 SDK 使用文档。
5. 整理 Windows / Linux 打包与发布流程。

---

*方案制定日期：2026-07-13*  
*最后更新：2026-07-20*
