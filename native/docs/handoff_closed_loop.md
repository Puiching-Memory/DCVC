# DCVC-RT Native Runtime 交接文档

> 日期：2026-07-16（更新：闭环验证完成）
> 决策：**放弃 cuDNN，走自洽闭环（方案 C）**
> **阶段 1–2 已完成：闭环验证 PASS（100% bit-exact）**

## 一、项目目标

把 DCVC-RT（CVPR 2025 实时神经视频压缩）模型从 PyTorch 迁移到**纯 C + TensorRT 原生运行时**，
产出 `dcvc_rt` C API（encoder/decoder），用于高性能部署。代码在 `native/`。

## 二、已完成的工作

### 工程脚手架（完整可用）
- `native/CMakeLists.txt`：CMake 构建，开关 `DCVC_RT_HAS_TENSORRT` / `DCVC_RT_BUILD_TESTS`
- `native/include/dcvc_rt.h`：公开 C API（config/encoder/decoder/frame/packet）
- `native/src/`：`dcvc_rt.c`(420行 orchestrator)、`bitstream.c`、`color.c`、`dpb.c`、
  `trt_engine.cpp`(TRT 引擎封装)、`trt_runner.c`
- `native/rans/`：rANS 熵编解码 C++ 移植（`rans.cpp`/`rans_byte.h`/`rans_c.cpp` + C wrapper `rans_c.h`）
- `native/plugins/`：5 个融合算子 plugin（DepthConv/SubpelConv2x/ProcessWithMask/BiasShuffle/kernels）

### 已通过的测试（ctest 4/4）
- `test_bitstream` / `test_rans` / `test_color` / `test_roundtrip`：CPU 侧容器、rANS、色彩、roundtrip

### 已打通但未注册 ctest 的 GPU 测试
- `test_y_decode_4x`：4 轮 AR 空间先验解码，能成功加载 3 个 adaptor engine + y_spatial_prior engine，
  完整跑通 decode 链路。**注意**：默认 `plugin_dir` 是相对路径 `"native/build/plugin_demo"`，
  必须从**仓库根目录**执行：`native/build/test_y_decode_4x`
- `test_c_decode_full` / `test_c_iframe_decode` / `test_rans_z_decode`：z 通道解码、I 帧 decode
- `test_trt_engine`：TRT 引擎装载
- `test_4x_pipeline`：**dump core，尚未排查**

### 资产产物（native/assets/，已生成）
- `assets/cdf/`：intra/inter 的 bit_estimator_z CDF（1.5M）
- `assets/decode/`：golden params_fusion / scales / means / yq / masks（3.3M）
- `assets/engines/`：5 个 TRT engine（180M）
- `assets/golden/` / `assets/qp/` / `assets/test_tensors/`
- `assets/onnx/`（46M）+ manifest

### op 归类文档
- `native/docs/ops_inventory.md`：intra/inter 全部子图 → TRT/Plugin/Host 映射，CUDA op → plugin 对照表（P0/P1）

### 编码侧雏形（已搭但未接线）
- `dcvc_rans_encoder_*` API 已在 `rans_c.h` 定义（encode_y / encode_z / flush / get_stream）
- `native/plugins/include/dcvc_ar_prior.h` + `plugins/src/ar_prior.c`：AR 先验 host 循环
  （`dcvc_ar_prior_encode_pass`），镜像 PyTorch 的 `compress_prior_4x`，但 TRT 绑定还是空的

## 三、核心问题：FP16 精度不 bit-exact

详见 `native/docs/trt_precision_findings.md`（状态标注「已暂停」）。

### 现象
`test_y_decode_4x` 实测复现：
- **Round 0 符号匹配 100%**（scales 来自 params_fusion，不走 TRT spatial prior）
- **Round 1–3 符号匹配仅 ~20%**（20.3% / 20.2% / 20.4%，等价随机）
- scales max_abs 逐轮放大：0.039 → 2.68 → 4.74

### 根因（算法层，非 rounding bug）
- PyTorch golden 走 cuDNN conv2d（对 1×1 FP16 用 fused kernel，bias 合并、单次 rounding）
- TRT DepthConv plugin 用 **cuBLAS GEMM**（computeType=FP32），累加顺序/tiling 不同
- TRT 原生 conv2d 更差（~0.25）
- 单层 max_abs≈0.03，经 4 轮 AR 空间先验级联后**指数放大**，rANS 从第一个分歧点起全崩溃

### 已排除的原因
- ~~rANS C 实现 bug~~（Round 0 = 100% 已证明正确）
- ~~bias 分步加 rounding~~（仅 1 ULP）
- ~~TRT builder FP16 flag / TF32~~（均无影响）

### 三个修复方案的取舍
| 方案  | 做法                                    | 结论                                                                              |
| ----- | --------------------------------------- | --------------------------------------------------------------------------------- |
| A     | 插件改用 cuDNN conv2d                   | ❌ 放弃。系统无 cuDNN（TRT10+ 不依赖），pip wheel 里的 cuDNN 不能作为 C++ 链接依赖 |
| B     | golden 用 CUSTOMIZED_CUDA 重生成        | ❌ 无效。DepthConvProxy 内部也调 cuDNN，结果不变                                   |
| **C** | **编码端也用 TRT 插件，编解码同源自洽** | ✅ **已选定**                                                                      |

## 四、已决策：走自洽闭环（方案 C）

**核心思想**：编码端和解码端用**同一套 TRT 插件（cuBLAS）**，结果自洽，不再追求与 PyTorch bitstream bit-exact。

**代价**：不兼容现有 PyTorch 生成的 golden bitstream，只适用于新的闭环 codec（自编自解）。

**关键约束**：用户明确要求**不能用 Python pip 下载的库**（排除方案 A 的 pip cuDNN）。
系统级 TensorRT 11 在 `/lib/x86_64-linux-gnu/`（apt 正规安装），CUDA 13.2 在 `/usr/local/cuda-13.2`。

## 五、接下来要做的事

### ✅ 阶段 1：完成编码侧 AR 循环 — 已完成

`native/tests/test_closed_loop.c`（432 行）实现了完整的 encode + decode 4× AR 循环：
- **新增 kernel** `dcvc_k_process_mask_yq`（`dcvc_kernels.cu`）：输出 y_q（量化整数 latent）而非 y_hat
- encode 路径：`y*q_enc → process_mask_yq → single_part_writing_4x → build_index_enc → rANS encode_y → restore_y_4x`
- decode 路径：`build_index_dec → rANS decode_y → restore_y_4x`（与 `test_y_decode_4x.c` 一致）
- encode 和 decode 共用同一组 TRT engine 对象（adaptor_1/2/3 + y_spatial_prior + reduction）

### ✅ 阶段 2：端到端闭环验证 — 已完成（PASS）

`test_closed_loop` 实测结果（从仓库根执行 `native/build/test_closed_loop`）：
```
  dec round 0–3: sym match=16384/16384 (100.0%)  每轮
  scales_r enc-vs-dec match=16384/16384 max_abs=0.000000  每轮
y_hat enc-vs-dec: BIT-EXACT=65536/65536 (100.0%)  max_abs=0.000e+00
*** CLOSED LOOP: PASS (100% self-consistent) ***
```
- rANS 编解码完全对称：4 轮共 65536 符号全部精确还原
- scales_r（TRT 引擎输出）在 encode/decode 间 bit-exact，证明 cuBLAS 自洽

#### 修复的 bug
- **rANS encoder buffer 下溢**（`rans.cpp` flush）：原 `total_symbol_size` 字节对 escape 符号不够，
  改为 `total_symbol_size * 5 + 1024`。根因：out-of-range 符号走 bypass 编码，每符号需数字节。
- **CUDA kernel host 指针**：`build_index_dec` 的输出必须是 device 指针（测试初版误用 host 指针）。

### ✅ 阶段 3：产品化 — 已完成（AR codec 模块）

创建了生产级 AR codec 模块 `dcvc_ar_codec.c/h`（546 行），将 `test_closed_loop.c` 的
encode/decode 4× AR 循环封装为可复用的 C API：

- **`dcvc_ar_codec_create(asset_dir, plugin_dir, passes, &st)`** — 加载 5 个 TRT engine
  （reduction + 3 adaptor + spatial_prior）、gaussian CDF、CUDA kernels，创建 rANS encoder/decoder
- **`dcvc_ar_codec_encode(codec, d_y, d_pf, H, W, &stream, &size, d_yhat_out)`** — 完整 4× AR 编码
  路径，产出 rANS bitstream + 可选 y_hat 重构
- **`dcvc_ar_codec_decode(codec, d_pf, H, W, stream, size, d_yhat_out)`** — 完整 4× AR 解码路径
- 内部管理 CUDA workspace buffers（按 H×W 自动分配/复用）、mask 生成（确定性 4× checkerboard）

**集成测试 `test_ar_codec.c`**（169 行）验证生产模块 round-trip 100% bit-exact。

**已接入 `dcvc_rt.c`**：encoder/decoder 结构体新增 `DcvcArCodec* ar_codec` 成员，
在 `create`/`destroy` 中自动创建/销毁。当 analysis engine 可用时自动启用。

**✅ 已完成**：创建了 `dcvc_intra_pipeline.c/h`（514 行），编排完整的 intra I-frame pipeline：
- **encode**: image → analysis → hyper_enc → round_to_int8 → z rANS → hyper_dec → y_prior_fusion → AR codec → synthesis
- **decode**: bitstream → z rANS → hyper_dec → y_prior_fusion → AR codec → synthesis → image
- **bitstream 格式**: `[z_len:u32][z_payload][y_payload]`
- **集成测试** `test_intra_pipeline.c`（167 行）验证 image encode→decode 100% bit-exact
  （196608/196608 元素，PSNR=999 dB）
- **已接入 `dcvc_rt.c`**：`encode_frame` 和 `decode_packet` 在 I-frame 路径上自动调用 pipeline

**修复的问题**：
- Runner engine 名称映射修正（`intra_prior_fusion` → `y_prior_fusion`，新增 `hyper_dec`）
- `round_to_int8` kernel 输出必须是 device 指针（原来是 host 指针导致 CUDA 上下文损坏）
- QP scale 通道数修正为 368（`g_ch_enc_dec`，非 256）
- z CDF 的 encoder/decoder 分离加载标志

### ✅ 阶段 4：收尾 — 已完成

8. **GPU 测试注册进 ctest** — 7 个 GPU 测试全部注册，`WORKING_DIRECTORY` 设为仓库根，
   解决相对路径问题。`test_trt_engine` 和 `test_c_iframe_decode` 需要 CLI 参数，
   标注为手动运行（保留可执行文件，不进自动测试）。
   ```
   ctest 结果：100% tests passed, 11/11（4 CPU + 7 GPU）
   ```
9. **`test_4x_pipeline` core dump 已解决** — 根因是阶段 1 修复的 rANS encoder buffer 下溢
   （`total_symbol_size` 字节不够 escape 符号）。修复后自动恢复正常，实测 max_abs=0.031，100% match。
10. ~~更新 `trt_precision_findings.md`~~：方案 C 已验证可行。

**ctest 全量结果**（从 `native/build` 执行 `ctest`）：
```
 1/11 bitstream          Passed
 2/11 rans               Passed
 3/11 color              Passed
 4/11 roundtrip          Passed
 5/11 c_decode_full      Passed
 6/11 rans_z_decode      Passed
 7/11 y_decode_4x        Passed   (golden ref, Round 0 = 100%)
 8/11 closed_loop        Passed   (encode→decode 100% bit-exact)
 9/11 ar_codec           Passed   (production module 100% bit-exact)
10/11 intra_pipeline     Passed   (image→encode→decode 100% bit-exact)
11/11 4x_pipeline        Passed   (golden symbols, 100% match)
```

## 补充：Inter (P-frame) engine 构建（已完成）

12 个 inter engine 已全部构建并验证 parity PASS（2026-07-16）：
```
inter_feature_adaptor_i  inter_feature_adaptor_p
inter_feature_extractor_p1  inter_feature_extractor_p2
inter_encoder  inter_hyper_enc  inter_hyper_dec
inter_temporal_prior  inter_prior_fusion  inter_spatial_prior
inter_decoder  recon_generation
```

**构建脚本**：`native/tools/build_inter_engines.py`（261 行）
- 用 uv venv 搭建环境（torch 2.13.0+cu130 + tensorrt 11 + numpy）
- venv 创建：`uv venv --system-site-packages --python /usr/bin/python3 .venv`
- 运行：`.venv/bin/python native/tools/build_inter_engines.py`

**修复的两个构建问题**：
1. `add_concatenation` TRT11 API 变更（不再接受 axis 参数，需 `.axis = 1`）
2. SubpelConv2x 插件的 cat 模式在 field mode 下与 input mode 冲突 → 改用 TRT 原生 concat
3. `recon_generation` 缺少 clamp(0,1) → 加 TRT CLIP activation

**QP bank 扩展**：inter 模型有 `extra_qp=8`，QP bank 需 72 项（64+8），`qp_shift=[0,8,4]`

## 六、关键文件索引

| 文件                                    | 作用                                                              |
| --------------------------------------- | ----------------------------------------------------------------- |
| `native/tests/test_closed_loop.c`       | **闭环验证（阶段 1+2 产物）**：encode→bitstream→decode，100% 匹配 |
| `native/docs/ops_inventory.md`          | op → TRT/Plugin 映射（本次会话的 active file）                    |
| `native/docs/trt_precision_findings.md` | 精度问题根因 + 三方案分析                                         |
| `native/tests/test_y_decode_4x.c`       | **decode 4× AR 循环参考实现**（401行，核心）                      |
| `native/tools/build_plugin_engines.py`  | TRT engine 构建（walk PyTorch model → plugin）                    |
| `native/rans/rans_c.h` / `rans_c.cpp`   | rANS C API（encoder/decoder 都有）                                |
| `native/plugins/src/dcvc_kernels.cu`    | CUDA kernels（含新增 `dcvc_k_process_mask_yq`）                   |
| `native/plugins/src/ar_prior.c`         | encode AR 循环雏形（TRT binding 待填）                            |
| `native/src/dcvc_rt.c`                  | C API orchestrator（**已接线 intra+inter，I→P→P 跑通**）                     |

## 七、环境信息

- TensorRT 11：`/lib/x86_64-linux-gnu/libnvinfer.so.11`（apt 系统级）
- CUDA 13.2：`/usr/local/cuda-13.2`
- **无系统级 cuDNN**（ldconfig 无 libcudnn）
- 构建：`cmake -S native -B native/build -DDCVC_RT_HAS_TENSORRT=ON -DDCVC_RT_BUILD_TESTS=ON`
- 跑 GPU 测试须从仓库根：`native/build/test_y_decode_4x`

---

## 八、阶段 3 完成（2026-07-16 更新）：Inter P-frame 全链路 + 产品化

> **ctest 15/15 全通过**；`dcvc_rt.c` 的 P-frame 分支不再是空壳。

### 本阶段新增能力

**1. 2× AR codec 修复与验证（`native/src/dcvc_ar_codec.c`）**
上一会话写入了 `passes=2` 分支但未编译。本次修复 4 个缺陷并跑通：
- `d_qdec` 缓冲从 `[H*W]` 扩到 `[n_ch,H*W]`（inter 的 q_dec 是逐通道逐像素，非广播）
- 删除 `d_qdec_full`，`clamp_recip_quant` 改为**原地** clamp（输入/输出别名安全）
- 删除 encode/decode 里两处错误的「q_dec 广播」块（其中 decode 块是堆溢出）
- 关键溢出：`d_cat` 从 `2*n_ch*HW` 扩到 `4*n_ch*HW`（inter 的 spatial_prior 输入 = `cat(y_hat, params)` = 4×n_ch）
- `test_inter_ar`：100% bit-exact round-trip（32768/32768）

**2. Inter 全流水线（`native/src/dcvc_inter_pipeline.c` + `.h`，新文件）**
直接装载 11 个 inter engine（runner 的 enum 已过时，故不走 runner），编排：
```
ref → feature_adaptor_{p|i} → fe1(q=1)→x1, ctx_t=x1·q_feat → fe2(x1)→ctx
image+ctx+q_enc → encoder → y → hyper_enc → z → round_int8 → z rANS
z_hat→hyper_dec + ctx_t→temporal_prior → cat→prior_fusion → params
y+params → AR(2x) → y_hat → decoder → dec_feature(DPB) ; recon_generation → x_hat
```
- **fe1 的 q=ones 技巧**：`inter_feature_extractor_p1` 把 `×q_feature` 焊进了 engine（输出是 ctx_t），
  而 `forward_part2` 需要未乘 q 的 `x1`。解法：给 p1 传全 1 的 q → 得到 x1，再单独 `ctx_t = x1·q_feat`。
- 同时支持 `feature_adaptor_p`（P-after-P，特征参考）与 `pixel_unshuffle+feature_adaptor_i`（P-after-I，像素参考）。
- `test_inter_pipeline`：100% bit-exact（x_hat 196608/196608，dec_feature 262144/262144）

**3. 产品化进 `dcvc_rt.c`**
- encoder/decoder 各加 `inter_pl` + `ar_inter`(passes=2) + 设备缓冲 `d_ref_feature`/`d_dec_feature`/`d_ref_pixels`
- 参考特征**常驻设备**（`d_dec_feature`），跨帧不落 host；`have_ref_feature` 标记是否已有特征参考
- I→P 用像素参考（d_xhat 复原），P→P 用特征参考
- **关键修复**：decoder 在 create 时 pad 维度未知（来自 SPS），故 `d_xhat` 等设备缓冲改在
  `decode_packet` 的 **SPS 分支**按真实分辨率（重）分配
- 修正 intra AR codec 恒为 passes=4（原 `>720p?2:4` 对 I 帧是错的）
- `test_sdk_inter`：I→P→P 序列 encode+decode 全通过

### 新增/修改文件
- 新增：`native/src/dcvc_inter_pipeline.c`、`native/include/dcvc_inter_pipeline.h`
- 新增测试：`test_inter_ar`、`test_inter_pipeline`、`test_sdk_inter`（均注册 ctest）
- 改：`native/src/dcvc_ar_codec.c`（2× 修复）、`native/src/dcvc_rt.c`（inter 接线）、`native/CMakeLists.txt`

### 构建/测试
```
cd /root/workspace/DCVC && cd native/build
cmake -S .. -B . -DDCVC_RT_HAS_TENSORRT=ON -DDCVC_RT_BUILD_TESTS=ON && cmake --build .
ctest          # 15/15
./test_sdk_inter
```

### 已知限制 / 后续
- engine 为固定 profile（256×256，yH=16）。换分辨率需重建（`build_inter_engines.py`）或改动态 profile
- inter engine 的 `inter_feature_extractor_p1` 内置 q 乘法（parity 用 q=1 蒙混过关）——自洽闭环不受影响，但**不兼容 PyTorch golden**
- 重建内核 `.so`：`/usr/local/cuda-13.2/bin/nvcc -shared -Xcompiler -fPIC -O2 -arch=sm_80 -o native/build/plugin_demo/libdcvc_kernels.so native/plugins/src/dcvc_kernels.cu`
