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


---

## 8. CUDA Graphs 探索结论（2026-07-17）：同样净零，已回退

继 shape/address 缓存（§7）无效后，尝试了 host-bound 假设下收益最大的方向：**CUDA Graphs**——把稳态 P 帧的 enqueue 序列录成图，一次 `cudaGraphLaunch` 替代多次 `enqueueV3`。

### 8.1 实施与验证（技术上成功）
- TRT 11.0 + CUDA 13.2 全支持图捕获。实现了 `GraphSeg` 机制（warmup→capture→replay 三态，失败自动降级为 raw）。
- 把 encode 的两个纯-GPU 段 E1（5 引擎：ref→fe1→fe2→enc→henc→round）和 E2（3 引擎：hdec→temp→pfus）做进图，分段边界是 rANS 的 D2H/sync host seam。
- 插桩确认：**capture 成功 + replay 生效**（`cudaGraphLaunch` 在 frame 3+ 触发），bit-exact、ctest 15/15 PASS。

### 8.2 性能：净零（实测，高帧 A/B）
| 变体（100 帧 × 5 次，e2e 中位） | 延迟 |
|---|---|
| 基线 | **20.26 ms** |
| 图化（E1+E2，确认 replay） | 20.58 ms |

图化版本**没有任何改善，反而略慢 ~0.3ms**（噪声内）。决定性否定。

### 8.3 为什么失败（两个根因，均已实测验证）
**A. host enqueue 本就不在关键路径上（与 GPU 重叠）**
shape-cache 那轮已证明 enqueueV3 是 ~6.6ms 的 host CPU 时间。但它是**与 GPU 执行重叠**的——host 排下一个 engine 的同时 GPU 在跑当前 engine。删掉它不减 wall-clock。§7 的 GPU 利用率 29% 的那个 12.8ms 空闲，**不是 enqueue 造成的，是 sync stall 造成的**（`dcvc_sync` 把 GPU 排空 + CPU rANS 期间 GPU 干等）。

**B. 段太小，图 launch 抵消不掉手动 enqueue**
CUDA Graphs 的优势在于**节点很多**（几十上百）时一次 launch 替代百次 enqueue。我们的段只有 3–5 个节点，`cudaGraphLaunch` 自身开销与 3–5 次 `enqueueV3` 相当，没有 launch-amortization 收益。

### 8.4 决策
图代码**全部回退**（无收益 + 增加复杂度/风险）。保留 shape/address 缓存（§7，正确且无害）。本环境的真正瓶颈是 **rANS sync stall**，而它只能靠「把 rANS 移出 CPU sync 路径」解决——但 §6 已证明 GPU rANS 净负面。所以在 256×256 + CPU rANS 架构下，host-side 优化（shape 缓存 / CUDA Graphs）均无法降低延迟。

### 8.5 复现命令
```bash
# 干净基线（shape-cache 保留，无 graph 代码）
cd /root/workspace/DCVC && cd native && cmake --build build -j8
CUDA_VISIBLE_DEVICES=0 ./native/build/test_perf 32 100    # e2e ~20.2ms
```


---

## 9. 决定性发现：原生比 PyTorch 慢 2×，根因是 sync 拓扑（不是 TRT）

### 9.1 实测对比（`bench_torch.py`，同分辨率/QP/帧/A30）
| | ENCODE | DECODE | E2E |
|---|---|---|---|
| **原生（当前）** | 10.9 ms | 9.4 ms | **20.3 ms** |
| **PyTorch 原版** | 5.4 ms | 4.9 ms | **10.3 ms** |
| 倍率 | 2.0× | 1.9× | **1.96×** |

原生全面慢 ~2×。PyTorch 比 TRT 快**不是 TRT 本身的问题**——是原生代码的流水线拓扑错了。

### 9.2 根因：rANS 处理方式（关键，颠覆本文 §6–§8）
两个版本的 rANS **都是 CPU 计算**（PyTorch 也是 `.cpu().numpy()`，见 `src/models/entropy_models.py:48,51`，rANS 库同源 `MLCodec_extensions_cpp`）。差异完全在**如何把 CPU rANS 与 GPU NN 推理重叠**：

**PyTorch（`video_model.py:299-338`）— event 驱动、GPU 几乎不空转：**
```
NN stream:   enc→henc→round_z →[EVENT z]→ params→AR(y) →[EVENT y]→ decoder→recon
rANS stream:                              wait(z_evt)→enc_z  wait(y_evt)→enc_y enc_y flush
            （NN 与 CPU-rANS 真并发；只在真实数据依赖处 event-wait）
最终：1 次 synchronize（结尾）
```

**原生（`dcvc_inter_pipeline.c` + `dcvc_ar_codec.c`）— 6 次 full-GPU drain：**
```
single stream: enc→henc→round_z
  ──dcvc_sync()── [排空 GPU] ── cudaMemcpy z→host ── CPU enc_z
  E2: hdec→temp→pfus
  ──dcvc_ar_codec: ──dcvc_sync()── [排空 GPU] ── packed→host ── CPU enc_y r0
                   ──dcvc_sync()── [排空 GPU] ── packed→host ── CPU enc_y r1   ... (decode 同理)
6× dcvc_sync = 6 次 GPU 完全排空 = 每次 CPU-rANS 期间 GPU 干等
```

### 9.3 推翻本文前面的错误结论
- **§6「z rANS 已异步并行、不在关键路径」**：对 z 编码成立（确实用 worker 线程重叠了），但 **AR 的 y-rANS 不是**——`dcvc_ar_codec.c` 里 4 处 `dcvc_sync()` 在 CPU rANS 前排空了 GPU。这是 ~2× 差距的主要来源。
- **§7 shape 缓存、§8 CUDA Graphs 无效**：都正确，但它们**治错了病**。真正的病是 sync stall，不是 enqueue 开销。enqueue 与 GPU 重叠（§8.3A），所以删掉它没用；sync 把 GPU 排空，才是延迟来源。

### 9.4 可行方向（治本，预期 ~2× 收益）
把原生改成 PyTorch 的 event 拓扑：CPU rANS 跑在独立高优先级 CUDA 流上，用 `cudaEvent` 在真实依赖处 wait，**不在 rANS 前 sync**。具体：
1. AR 的 `dcvc_sync()+D2H` 改成 `cudaMemcpyAsync` + record event；rANS 在等待 event 的同时，GPU 继续跑下一个独立的 NN engine。
2. 用专用 rANS 流（`priority=-1`），与 NN 流并发。
3. 全帧只在最后 1 次 `synchronize` 收尾。

这是唯一能追平 PyTorch 的方向。**不是 GPU rANS（§6 已证伪），而是 CPU rANS 的正确异步化。** 之前的 host-side 微优化（shape/graph）应全部放弃，集中做这个。

### 9.5 复现命令
```bash
cd /root/workspace/DCVC
.venv/bin/python bench_torch.py 32 50          # PyTorch: 10.3ms e2e
CUDA_VISIBLE_DEVICES=0 ./native/build/test_perf 32 100   # 原生: 20.3ms e2e
```
