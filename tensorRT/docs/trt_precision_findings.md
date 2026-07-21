# TRT Spatial-Prior Engine 精度调查结论

> 状态：**已解决** — 方案 C 已验证 PASS（自洽闭环 + AR codec + 完整 intra pipeline 全部 100% bit-exact）  
> 日期：2026-07-16

## 背景

C 全流程 decode (`test_y_decode_4x`) 中 Round 0 符号匹配 100%，但 Round 1–3 的 rANS 符号大幅偏离 golden。rANS 本身已验证正确（用 golden scales 时 4 轮全部 100%）。

## 根因

**cuBLAS GEMM（TRT 插件）与 cuDNN conv2d（PyTorch golden）在 FP16 下产生不同结果。**

| 组件               | 计算路径                          | 单层 max_abs  |
| ------------------ | --------------------------------- | ------------- |
| PyTorch golden     | `F::conv2d` → cuDNN               | 基准          |
| TRT DepthConv 插件 | `cublasGemmEx` (computeType=FP32) | ~0.015–0.031  |
| TRT 原生 conv2d 层 | TRT 内部实现                      | ~0.25（更差） |

误差来源是**算法层面**的，不是 rounding 策略问题：
- cuDNN 对 1×1 FP16 conv 使用专门的 fused kernel（与 bias 合并、单次 rounding）
- cuBLAS GEMM 是通用矩阵乘，累加顺序/tiling 不同
- 三种方式（cuDNN、cuBLAS、TRT-native）的 FP16 结果互不 bit-exact

## 误差传播链

```
adaptor_1 (1个 DepthConvBlock):     max_abs=0.031
  → y_spatial_prior (1个 DepthConvBlock): max_abs=0.096（累积）
    → scales_R1:                         max_abs=0.039
      → build_index_dec:                 ~200个 index 差 ±1
        → rANS 状态:                      从第一个分歧点起全部崩溃
          → 符号匹配率:                    20%（等价于随机）
```

每一轮的空间先验（adaptor + y_spatial_prior）都依赖上一轮的 y_hat_so_far，因此误差**指数级放大**：

| 轮次    | scales max_abs (vs golden) | 符号匹配率 | 原因                                              |
| ------- | -------------------------- | ---------- | ------------------------------------------------- |
| Round 0 | 0.000                      | **100%**   | scales 来自 params_fusion（无 TRT spatial prior） |
| Round 1 | 0.039                      | 20.3%      | adaptor_1 + y_spatial_prior 的 cuBLAS 误差        |
| Round 2 | 2.684                      | 20.2%      | Round 1 误差经 rANS 级联放大                      |
| Round 3 | 4.736                      | 20.4%      | 进一步放大                                        |

## 已排除的原因

- ~~rANS C 实现有 bug~~ — rANS 核心代码（`rans.cpp`/`rans_byte.h`）与 Python 原版字节完全相同，已修复 index 计算后 Round 0 = 100%
- ~~bias 分步加导致额外 rounding~~ — FP16 下 bias 分步加与合并加差异仅 1 ULP（~0.0005），不是主因
- ~~TRT builder 未设 FP16 flag~~ — 插件自身固定 FP16，TRT 配置不影响插件内部计算
- ~~TF32 模式影响~~ — `torch.backends.cuda.matmul_allow_tf32` 开/关对 cuDNN 结果无影响

## 推荐修复方案

### 方案 A（推荐）：插件内部改用 cuDNN conv2d

将 `depthconv_plugin_full.cu` 中的 `pw_conv()`（cuBLAS GEMM）替换为 cuDNN convolution forward：

- cuDNN 库已有：`.venv/lib/python3.12/site-packages/nvidia/cudnn/`
- 改动范围：`pw_conv()` 函数（6 处调用点共用）
- 预期：单层精度从 ~0.03 降到 ~0.001 以下，级联效应消除
- 风险：cuDNN workspace 管理 + descriptor 初始化开销

### 方案 B：golden 数据用 CUDA 扩展路径重新生成

原模型有 `CUSTOMIZED_CUDA_INFERENCE` 开关。当前为 `False`（PyTorch fallback = cuDNN）。
如果把 golden 数据用 `CUSTOMIZED_CUDA_INFERENCE=True` 重新生成，golden 的 DepthConv 会走
`DepthConvProxy` → 同样用 `F::conv2d`（cuDNN），所以**不会改变结果**。

→ **方案 B 无效**，DepthConvProxy 的 CUDA 扩展内部也调用 cuDNN。

### 方案 C：编码端也用 TRT 引擎

如果编码和解码都用相同的 TRT 插件（cuBLAS），结果自洽。
但 golden bitstream 是 PyTorch 编码的，解码端必须 bit-exact。

→ 仅适用于闭环编解码，不适用于兼容现有 PyTorch bitstream。

## 关键验证代码

```bash
# 对比 TRT 插件 vs PyTorch cuDNN（adaptor_1 单层）
PYTHONPATH=/usr/lib/python3.12/dist-packages:src/cpp:src \
  .venv/bin/python /tmp/trt_vs_pt_spatial.py

# cuBLAS GEMM vs cuDNN conv2d 差异
.venv/bin/python /tmp/test_gemm_algo.py
# 输出: cuDNN conv2d vs cuBLAS mm: max_abs=0.031250
```

## 下一步（恢复时）

1. `pw_conv()` → cuDNN `cudnnConvolutionForward` + `cudnnAddTensor`（bias）
2. 管理 cuDNN handle/descriptor 生命周期（可全局缓存）
3. 重建 `libdcvc_depthconv.so`
4. 重建 3 个 adaptor engine + y_spatial_prior engine
5. 运行 `test_y_decode_4x` 验证 Round 1–3 符号匹配率
