# GPU rANS 探索结论与决策记录

> 日期：2026-07-17
> 结论：**y 与 z 的 rANS 上 GPU 均为净负面，已全部回退到 CPU rANS（Phase 0+2 基线）。**
> 本文档保留探索过程与判定依据，供后续判断是否复访该方向时参考。

---

## 结论（TL;DR）

| 方向         | 判定                         | 依据                                                 |
| ------------ | ---------------------------- | ---------------------------------------------------- |
| y rANS → GPU | **净负面（实测）**           | BPP +13%、decode 更慢、GPU rANS 占 54% GPU 时间      |
| z rANS → GPU | **净负面（判定，含 1080p）** | z rANS 已在 worker 线程与 GPU 并行，本就不在关键路径 |

决策：保持 CPU rANS。真正瓶颈是 TRT 推理与 `enqueueV3` 的 host 开销，不是 rANS。失败本质是结构性的——把「已能和 GPU 并行的 CPU 工作」或「过小的数据量」搬上 GPU，必然净亏。

---

## 1. 基线

DCVC-RT native 运行时，纯 C + TensorRT + CUDA kernel，无 pip 依赖。y 与 z 均用 CPU rANS（`native/rans/rans.cpp`，`RansEncoder/DecoderLibMultiThread`，**编/解码各有一条 `std::thread`**——这一点对 §6 的判定是决定性的）。

基线指标（256×256，QP=32，A30）：

| 指标   | 值        |
| ------ | --------- |
| BPP    | **0.046** |
| PSNR_Y | 34.94 dB  |
| decode | 8.7 ms    |
| encode | 9.5 ms    |

---

## 2. 尝试：y rANS 上 GPU

采用 dietgpu 风格的交错 rANS，K 个独立 32-bit 状态并行。K=4 是「BPP header 开销」与「并行度」的折中。

- bitstream 格式：`uint16 K | uint16 N | uint16 dataBytes[K] | uint32 finalStates[K] | data[]`
- 自洽性：encode 与 decode 共享同一格式，bit-exact round-trip，ctest 15/15 PASS。

---

## 3. 结果：净负面（实测）

| 指标     | CPU rANS | GPU rANS  | 差值                |
| -------- | -------- | --------- | ------------------- |
| BPP      | 0.046    | **0.052** | +13%（header 开销） |
| decode   | 8.7 ms   | 9.0 ms    | +0.3 ms（更慢）     |
| encode   | 9.5 ms   | 9.8 ms    | +0.3 ms（更慢）     |
| 每帧字节 | ~251     | ~303      | +52 B               |

没有任何性能收益，BPP 反而更差。

---

## 4. nsys 证据：GPU rANS 占 54% GPU 时间

每帧 decode+encode 的 kernel 统计（8 帧）：

| 内核                              | 每次耗时 | 每帧次数 | 占全部 GPU 时间 |
| --------------------------------- | -------- | -------- | --------------- |
| `gpu_rans_decode_kernel`          | 2.03 ms  | 2        | **27.5%**       |
| `gpu_rans_encode_kernel`          | 1.97 ms  | 2        | **26.6%**       |
| 全部 TRT 推理（GEMM+conv+plugin） | —        | —        | 34%             |
| 其他 CUDA kernel                  | —        | —        | 12%             |

GPU rANS 吃掉 54% 的 GPU 时间，是全系统第一大瓶颈——比所有神经网络推理加起来还多。

---

## 5. 三个结构性失败原因

**A. GPU 整数除法是性能杀手**
rANS 核心是 `r = (r/freq)*65536 + (r%freq) + start`。GPU 无硬件整数除法，靠软件模拟，比 CPU 慢 ~10×。K=4 个线程各串行处理符号，每符号 2 次 int div → 单线程 ~2 ms。
（若一定要在 GPU 做 rANS，可考虑用 `__frcp_rn` 浮点倒数近似替代精确整数除法，或查表法——均未验证精度。）

**B. y 数据量太小（16K 符号），K 进退两难**
- K 小（=4）：并行度不足，2 ms/轮 ≈ CPU 20 倍
- K 大（=64）：每状态 `finalState 4B + dataOffset 2B = 6B`，64 状态 × 2 轮 = 768 B header，加在 ~300 B 的 bitstream 上直接翻倍 → BPP 从 0.046 爆炸到 0.161

y 每轮只有 8192–16384 符号，无法同时满足「K 足够大喂饱 GPU」和「K 足够小不爆 BPP」。

**C. 被迫加 dcvc_sync() 抵消了收益**
TRT `enqueueV3` 内部用默认流（stream 0），与 `dcvc_stream()`（non-blocking）上的后续 GPU kernel 跨流竞争。CPU rANS 版本在 D2H 前的隐式 sync 掩盖了此问题；GPU rANS 版本必须在每轮 `restore_y2x` 前显式 `dcvc_sync()` 才能保证正确性，把想消除的同步点又加回来了。

---

## 6. z rANS 上 GPU 的可行性再分析（关键修正）

### 6.1 早期假设（已被证伪）
早期假设 z 符号数 = 192 × (H/8)² = 196,608，预期省 ~2 ms。

### 6.2 实际符号数
z 符号数 = `ZC(128) × (H/64)²`（不是 (H/8)²）：

| 分辨率            | z 符号数                              |
| ----------------- | ------------------------------------- |
| 256×256           | **2,048**（比 y 每轮 16,384 还少 8×） |
| 1080p (1920×1080) | ~60,000                               |
| 4K (3840×2160)    | ~250,000                              |

### 6.3 决定性事实：z rANS 已异步并行，不在关键路径
- `native/rans/rans.h:128,192` + `rans.cpp:264,444`：`RansEncoder/DecoderLibMultiThread` 各有一条 `std::thread`，`encode_z/decode_z` 投递任务后**立即返回**，结果在后续取。
- 重叠窗口内**无 hidden sync**（已 grep 确认）：
  - decode 重叠窗 `build_ref_feature → fe1 → TEMP`（`dcvc_inter_pipeline.c` ~L531–545）：0 个 `dcvc_sync`
  - encode 重叠窗 `HDEC → TEMP → pfus → AR → dec → recon`（~L443–476）：0 个 `dcvc_sync`
- 256×256 实测（`clock_gettime` 插桩，稳态 6 帧）：
  - decode 整个 rANS 区段 0.35 ms，其中**未重叠尾段（host 阻塞）仅 0.09 ms**
  - encode 到 mux 时未重叠 = **0.00 ms**（完全被 GPU 盖住）

### 6.4 1080p 关键路径
首个 z_hat 消费者（HDEC）能启动的时间 = `max(GPU 重叠窗完成, z rANS 完成)`：

| 环节                                             | 1080p 量级 | 在关键路径? |
| ------------------------------------------------ | ---------- | ----------- |
| z rANS（worker 单线程）                          | ~2–4 ms    | 否          |
| decode 重叠窗 GPU（fe1+TEMP）                    | ~10–30 ms  | 是          |
| encode 重叠窗 GPU（HDEC+TEMP+pfus+AR+dec+recon） | ~15–40 ms  | 是          |

1080p 下 GPU 重叠窗（10–40 ms）远大于 z rANS（2–4 ms）→ z rANS 早就跑完在那等着。

### 6.5 结论
即便在 1080p，z 上 GPU：decode 临界路径省时 ≈ **0 ms**，反而新增 TRT 跨流 `dcvc_sync()`（同 §5-C 坑）+ header BPP。**净负面，与 y 同构。**

---

## 7. 最终状态

- y / z 均回退 CPU rANS，恢复 0.046 BPP 基线，ctest 15/15 PASS。
- GPU rANS 的实现文件已删除，`CMakeLists.txt` 已回退，无任何 rANS-on-GPU 残留。
- 真正瓶颈是 TRT 推理（GPU 端 GEMM/conv kernel ~6 ms/帧）与 `enqueueV3` 的 host 开销（~6 ms/帧）。

### 复现命令
```bash
cd /root/workspace/DCVC
cmake -S native -B native/build -DDCVC_RT_HAS_TENSORRT=ON -DDCVC_RT_HAS_CUDA=ON -DDCVC_RT_BUILD_TESTS=ON
cmake --build native/build -j8
CUDA_VISIBLE_DEVICES=0 ./native/build/test_quality 32          # 质量+BPP
CUDA_VISIBLE_DEVICES=0 ./native/build/test_perf 32 8           # 性能
cd native && ctest --test-dir build --output-on-failure         # 全量 15 项
```
