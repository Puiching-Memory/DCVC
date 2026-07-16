# DCVC-RT 定制推理引擎方案

> 基于成熟推理引擎（ONNX Runtime + TensorRT EP）的二次开发方案，面向 NVIDIA 设备。

---

## 1. 设计目标

- 为 DCVC-RT 提供独立的 C 语言推理 SDK，脱离 Python / PyTorch 运行环境。
- 支持 NVIDIA GPU 上的实时 / 近实时视频编码与解码。
- 保留 DCVC-RT 的宽码率、率控、YUV/RGB 统一等核心能力。
- 利用成熟推理引擎降低开发周期与维护成本。

---

## 2. 技术选型

| 层级         | 选型                                     | 原因                                        |
| ------------ | ---------------------------------------- | ------------------------------------------- |
| 基础运行时   | ONNX Runtime (C API)                     | 生态成熟、跨平台、支持自定义算子            |
| NVIDIA 加速  | TensorRT Execution Provider              | 自动图优化、CUDA Graph、NVIDIA 最优性能     |
| 自定义算子   | ONNX Runtime Custom CUDA Op              | 一次实现，ORT 原生与 TRT EP 均可调用        |
| 熵编码       | 纯 C rANS（复用 DCVC `src/cpp/py_rans`） | 脱离 Python / pybind11，保持 bitstream 兼容 |
| 数据流与状态 | 自研 C                                   | 帧间状态、内存池、GOP 管理                  |
| 对外接口     | 纯 C API                                 | 最大可移植性，便于集成到 C/C++/Go/Rust 等   |

---

## 3. 整体架构

```
┌─────────────────────────────────────────┐
│         dcvc_rt SDK (C API)             │
│  dcvc_rt_init / encode_frame / decode_frame
├─────────────────────────────────────────┤
│      Pipeline (encoder.c / decoder.c)   │
│  GOP 管理、帧间状态、量化参数、时序上下文  │
├─────────────────────────────────────────┤
│      Entropy Codec (rANS C)             │
│  PMF→CDF、算术编码、bitstream 读写        │
├─────────────────────────────────────────┤
│      ONNX Runtime (C API)               │
│  模型加载、Session Run、TensorRT EP 加速   │
│  Custom CUDA Op 插件                     │
├─────────────────────────────────────────┤
│      CUDA / cuDNN / TensorRT            │
└─────────────────────────────────────────┘
```

---

## 4. 项目目录结构

```
dcvc_rt_sdk/
├── 3rd_party/
│   ├── onnxruntime/                    # git submodule
│   └── tensorrt_plugin_ep/             # 可选，用于 TRT 扩展
├── src/
│   ├── dcvc_rt.h                       # 公共 C API
│   ├── dcvc_rt.c                       # 引擎封装、生命周期
│   ├── onnx/
│   │   ├── model_exporter.py           # PyTorch → ONNX 转换脚本
│   │   ├── custom_op_library.cu        # 自定义 CUDA 算子
│   │   └── custom_op_register.cc       # ORT 算子注册
│   ├── entropy/
│   │   ├── rans.c / rans.h             # rANS 编码/解码（纯 C）
│   │   ├── bitstream.c / bitstream.h   # 比特流读写
│   │   └── gaussian.c                  # 高斯参数估计
│   ├── pipeline/
│   │   ├── encoder.c / encoder.h
│   │   └── decoder.c / decoder.h
│   └── utils/
│       ├── tensor.c / tensor.h         # 张量抽象
│       ├── memory_pool.c               # 内存池
│       └── yuv.cu                      # YUV/RGB 转换（CUDA）
├── models/
│   ├── dcvc_rt_i.onnx                  # I帧模型
│   └── dcvc_rt_p.onnx                  # P帧模型
├── tests/
│   ├── test_op_parity.c
│   ├── test_i_frame.c
│   └── test_p_frame.c
├── examples/
│   ├── encode_yuv420.c
│   └── decode_yuv420.c
├── CMakeLists.txt
└── README.md
```

---

## 5. 关键算子与 ONNX 映射

| 算子                    | ONNX 原生              | 处理方式           |
| ----------------------- | ---------------------- | ------------------ |
| Conv2d                  | 是                     | 直接导出           |
| DepthwiseConv2d         | 是 (group=ch)          | 直接导出           |
| PixelShuffle            | 是                     | 直接导出           |
| PixelUnshuffle          | 是 (Reshape+Transpose) | 直接导出           |
| Sigmoid / ReLU          | 是                     | 直接导出           |
| WSiLU = x * sigmoid(4x) | 是                     | Sigmoid + Mul 组合 |
| SubpelConv2x            | 否                     | 自定义 CUDA 插件   |
| DepthConvBlock 融合     | 否                     | 自定义 CUDA 插件   |
| process_with_mask       | 否                     | 自定义 CUDA 插件   |
| bias_pixel_shuffle_8    | 否                     | 自定义 CUDA 插件   |
| build_index_enc / dec   | 否                     | 自定义 CUDA 插件   |

---

## 6. C API 设计草案

```c
#ifndef DCVC_RT_H
#define DCVC_RT_H

#include <stdint.h>
#include <stddef.h>

typedef struct dcvc_rt_context* dcvc_rt_ctx_t;
typedef struct dcvc_rt_stream*  dcvc_rt_stream_t;
typedef struct dcvc_rt_frame*   dcvc_rt_frame_t;

typedef enum {
    DCVC_RT_YUV420 = 0,
    DCVC_RT_RGB888 = 1,
} dcvc_rt_color_format_t;

typedef enum {
    DCVC_RT_SUCCESS = 0,
    DCVC_RT_ERROR_INVALID_ARG = -1,
    DCVC_RT_ERROR_OUT_OF_MEMORY = -2,
    DCVC_RT_ERROR_MODEL_LOAD = -3,
    DCVC_RT_ERROR_INFERENCE = -4,
    DCVC_RT_ERROR_ENTROPY = -5,
} dcvc_rt_status_t;

typedef struct {
    int width;
    int height;
    int qp;                               // 0-63
    dcvc_rt_color_format_t format;
    int intra_period;                     // -1 = all intra
    int rate_num;                         // 2-64
    int device_id;                        // GPU id
    int use_fp16;                         // 0/1
    int use_int16;                        // 0/1，预留跨设备确定性
    const char* model_path_i;             // I-frame ONNX
    const char* model_path_p;             // P-frame ONNX
} dcvc_rt_config_t;

int dcvc_rt_init(const dcvc_rt_config_t* config, dcvc_rt_ctx_t* ctx);
int dcvc_rt_destroy(dcvc_rt_ctx_t ctx);

int dcvc_rt_frame_create(int width, int height, dcvc_rt_color_format_t fmt, dcvc_rt_frame_t* frame);
int dcvc_rt_frame_destroy(dcvc_rt_frame_t frame);
uint8_t* dcvc_rt_frame_data(dcvc_rt_frame_t frame);

int dcvc_rt_encode_frame(dcvc_rt_ctx_t ctx, const dcvc_rt_frame_t frame, dcvc_rt_stream_t* stream);
int dcvc_rt_decode_frame(dcvc_rt_ctx_t ctx, dcvc_rt_stream_t stream, dcvc_rt_frame_t frame);

int dcvc_rt_stream_create(dcvc_rt_stream_t* stream);
int dcvc_rt_stream_destroy(dcvc_rt_stream_t stream);
size_t dcvc_rt_stream_size(dcvc_rt_stream_t stream);
const uint8_t* dcvc_rt_stream_data(dcvc_rt_stream_t stream);

const char* dcvc_rt_error_string(int status);

#endif
```

---

## 7. 开发阶段

### Phase 1：环境搭建与模型导出（2 周）

- 引入 ONNX Runtime 作为 git submodule，完成 CMake 构建。
- 将 DCVC-RT 的 `IntraEncoder / IntraDecoder / Encoder / Decoder` 导出为 ONNX。
- 验证 ONNX 输出与 PyTorch 输出的数值一致性（parity test）。

### Phase 2：自定义 CUDA 算子（3 周）

- 搭建 ONNX Runtime Custom Op 框架。
- 实现 `SubpelConv2x`、`DepthConvBlock` 融合、`process_with_mask`、`bias_pixel_shuffle_8`、`build_index_enc/dec` 等 CUDA 插件。
- 与 PyTorch 参考实现进行逐算子 parity 验证。

### Phase 3：熵编码与 Pipeline（3 周）

- 将 DCVC `src/cpp/py_rans` 改造为纯 C 库，移除 pybind11 依赖。
- 实现 bitstream 读写格式，保持与官方兼容。
- 实现 I 帧与 P 帧的完整编码/解码 pipeline，包含时序上下文管理。

### Phase 4：C API 封装与优化（2 周）

- 实现 `dcvc_rt.h` 中定义的 C API。
- 引入内存池、多 CUDA Stream、CUDA Graph 等优化。
- 在 NVIDIA GPU 上完成 1080p 编解码性能基线测试。
- 提供示例程序与基础文档。

---

## 8. 与全自研方案的对比

| 维度        | 全自研纯 C     | ONNX Runtime + TensorRT EP    |
| ----------- | -------------- | ----------------------------- |
| 开发周期    | 6-9 个月       | 2-3 个月                      |
| NVIDIA 性能 | 取决于手写内核 | 接近最优（TensorRT 自动优化） |
| 跨平台      | 最好           | 较好（ORT 支持多平台）        |
| 维护成本    | 高             | 低                            |
| 自定义空间  | 最大           | 中等（关键算子可插件化）      |
| 二进制大小  | 最小           | 较大（ORT 库）                |

---

## 9. 风险与缓解

| 风险                           | 缓解措施                                |
| ------------------------------ | --------------------------------------- |
| ONNX 无法完整表达某些融合算子  | 使用 Custom CUDA Op 实现                |
| TensorRT 对动态 shape 支持有限 | 固定 batch=1，使用动态高宽（ORT 支持）  |
| 自定义算子与 TRT EP 不兼容     | 先在 CUDA EP 上验证，再适配 TRT EP      |
| INT16 确定性推理               | 作为二期目标，先保证 FP16/FP32 功能正确 |
| 熵编码 bitstream 兼容          | 与官方 Python 版本进行 round-trip 对比  |

---

## 10. 下一步行动

1. 确认方案并创建项目目录结构。
2. 添加 ONNX Runtime 作为 git submodule。
3. 编写 ONNX 导出脚本，导出 I 帧与 P 帧模型。
4. 建立 parity test 框架，验证导出正确性。

---

*方案制定日期：2026-07-13*
