# 2026-06-24 Genllm CPU-only 推理测试对比总结

## 今日目标

对比本地 `Genllm` 推理系统与 `llama.cpp` 在同一模型、同一机器、CPU-only 模式下的推理性能，观察当前系统能力、主要瓶颈和后续优化顺序。

## 测试环境

- 机器：Intel(R) Core(TM) Ultra 7 265K
- CPU 核心/线程：20 physical cores / 20 logical threads
- 内存：约 47.4 GB
- 对比基线：`E:\model\llama.cpp`
- Genllm 工程：`G:\我的云端硬盘\Genllm`
- 模型权重：`E:\model\Qwen3-0.6B-IQ4_NL.gguf`
- 模型信息：Qwen3 0.6B, IQ4_NL, 约 596M 参数, 模型文件约 375.6 MB
- Genllm 构建配置：`cpu-release`, `BACKEND_CPU=ON`, GPU backend 全部关闭

## 执行命令

```powershell
cmake --build --preset cpu-release

& 'E:\model\llama.cpp\llama-bench.exe' `
  -m 'E:\model\Qwen3-0.6B-IQ4_NL.gguf' `
  -ngl 0 -t 20 -p 512 -n 128 -r 3 -o csv

& 'E:\model\llama.cpp\llama-bench.exe' `
  -m 'E:\model\Qwen3-0.6B-IQ4_NL.gguf' `
  -ngl 0 -t 20 -p 12,13,18,20 -n 32 -r 3 -o csv

& 'G:\我的云端硬盘\Genllm\build\cpu-release\bin\benchmark.exe' `
  'E:\model\Qwen3-0.6B-IQ4_NL.gguf' 32
```

`cmake --build --preset cpu-release` 返回 `ninja: no work to do`，说明 CLion/CMake 当前 CPU Release 产物已经是最新构建。

## 测试结果

### llama.cpp CPU-only

`llama-bench --list-devices` 显示没有 GPU 设备，实际加载的是 CPU backend：

- `ggml-cpu-alderlake.dll`
- `n_gpu_layers=0`
- backend=`CPU`

| 项目 | 配置 | 结果 |
|---|---:|---:|
| prompt processing | `n_prompt=512` | 约 1346.8 tok/s |
| decode | `n_gen=128` | 约 107.0 tok/s |
| 短 prompt prefill | `n_prompt=12/13/18/20` | 约 189.8 / 239.0 / 270.9 / 348.4 tok/s |
| 短生成 decode | `n_gen=32` | 约 31.0 tok/s |

### Genllm CPU-only

Genllm 识别到 CPU 设备，并将全部层分配到 CPU：

- CPU: `L:-1 ~ L:28`
- weights: 约 358.2 MB used / 479.9 MB reserved
- activation pool: 约 627.4 MB reserved
- KV cache: 约 224.0 MB used / 235.2 MB reserved

`benchmark.exe` 生成 32 token 的结果：

| prompt | prompt tokens | generated tokens | elapsed ms | ms/token | tokens/s |
|---|---:|---:|---:|---:|---:|
| 数学：`1+1=` | 12 | 32 | 11188.7 | 349.65 | 2.9 |
| 常识：中国首都是哪里？ | 13 | 32 | 10084.4 | 315.14 | 3.2 |
| 代码：prime 函数 | 20 | 32 | 13217.6 | 413.05 | 2.4 |
| 翻译：Hello, how are you? | 18 | 32 | 13310.4 | 415.95 | 2.4 |

## 对比结论

同一模型、同一 CPU-only 条件下：

- llama.cpp 32-token decode 约 31 tok/s。
- Genllm 32-token 端到端生成约 2.4-3.2 tok/s。
- Genllm 当前 CPU decode 大约比 llama.cpp 慢 10-13 倍。
- 如果参考 llama.cpp 128-token sustained decode 的约 107 tok/s，Genllm 与成熟 CPU 后端的差距可能达到 30 倍以上。

这个结果说明 Genllm 的图构建、GGUF 解析、KV cache、CPU-only 执行链路已经跑通，但 CPU 后端还处在功能正确优先阶段，性能内核还没有接近成熟推理库水平。

## 当前系统能力

1. 支持直接读取 GGUF 权重并识别 Qwen3-0.6B IQ4_NL 模型结构。
2. 支持 CPU-only 模式完成完整生成链路。
3. 调度器可以把 embedding、28 个 transformer layer、final norm/logits 全部分配到 CPU。
4. KV cache 工作正常，decode 阶段能够复用缓存。
5. benchmark 入口可用于回归测试，但目前指标把 prefill、decode、sampling 混在一起，后续需要拆分。

## 主要瓶颈判断

### 1. CPU 量化 linear/GEMV 是最大瓶颈

Genllm 的 IQ4_NL linear 路径当前做法是：

1. 每个输出列块读取量化权重。
2. 先反量化到临时 `bfloat16_t w_buf[BN][BK]`。
3. 再用 BF16 -> FP32 转换做 dot product。

这会带来三类成本：

- 重复反量化写临时 buffer。
- BF16 临时数据增加内存流量。
- decode 阶段 `M=1` 时不能充分利用通用矩阵分块。

llama.cpp 的优势在于 fused quantized GEMV：量化 nibble 解码、scale 应用、向量化 dot product 在同一条内核路径里完成，避免了大量中间写回。

### 2. CPU attention 仍是标量路径

Paged attention 当前是 head/token/dim 多层标量循环，且每个 head 创建临时 `std::vector<float>` 输出缓冲。短上下文时 linear 更重，但上下文变长后 attention 会迅速变成明显瓶颈。

### 3. sampling 会放大小模型 decode 开销

当前 `sample()` 对 151936 vocab 做 softmax，并在 top-p 时排序整个 vocab。对 0.6B 小模型来说，采样本身会占据可见比例。benchmark 如果目的是测后端算子，应提供 greedy/argmax 模式，避免采样策略污染内核性能判断。

### 4. benchmark 口径还不够细

Genllm 当前 benchmark 是 `executor.generate()` 整体计时，包含：

- prompt prefill
- 每 token decode
- sampling
- tokenizer 相关准备
- 可能的首次路径开销

因此现在只能做端到端体验对比，还不能精确定位每个阶段的耗时占比。

### 5. KV cache 池预留偏紧

本次 `max_seq_len=2048` 下 KV cache 显示 `224.0/235.2 MB`，使用率约 95.2%。这说明当前内存池估算可运行，但余量不大；未来扩大上下文、batch 或模型规模时需要重新审视 KV cache pool factor 和分配策略。

## 优化顺序建议

### P0：建立稳定、可拆分的 benchmark

优先级最高。没有清晰计时口径，后续优化容易误判。

建议新增：

- prefill 单独计时：`prompt_tokens / prefill_time`
- decode 单独计时：去掉首 token 后统计平均 `tok/s`
- sampling 单独计时：softmax/top-p/argmax 分开
- 每层耗时统计：至少记录 linear、attention、rms_norm、rope、silu、add/mul
- greedy benchmark：直接使用 argmax 或固定 token，排除 top-p 排序成本

### P1：为 IQ4_NL 实现 fused CPU GEMV

这是最关键优化点。

建议目标：

- 不再把 IQ4_NL 整块反量化成 BF16 buffer。
- 直接读取 4-bit nibble，查 `kvalues_iq4nl`，乘 block scale 后进入 FP32 accumulator。
- 针对 `M=1` decode 写专用 kernel。
- 输出列维度并行时减少 OpenMP 小任务开销。
- 对常见维度如 hidden=1024、intermediate=3072/4864、vocab=151936 做专门路径。

预期收益最大，应该先用微基准对单个 linear 层做 A/B 测试，再接入整图。

### P2：优化 logits 投影和 sampling

Qwen3-0.6B 的 vocab 约 151936，最后 logits projection 很重。

建议：

- benchmark 模式支持 `temperature=0` 或强制 argmax。
- top-p 不全量 sort，改用 partial selection 或 top-k 候选后再 top-p。
- logits projection 如果只为采样服务，可探索分块求最大值/top-k，减少完整 materialize 的必要性。

### P3：优化 CPU paged attention

建议：

- 复用 per-thread scratch buffer，避免每个 head 分配 `std::vector<float>`。
- head_dim=128 写 AVX2/FMA dot kernel。
- K/V cache 排布按连续读取优化，减少 page/block 指针跳转。
- decode 阶段使用单 token attention 专用路径。

### P4：降低执行器和内存池开销

建议：

- 检查每个 token 是否有不必要的 shape resolve、activation reset、临时分配。
- 固化 decode 图中不会变化的 tensor 元信息。
- 对常用 op 做 lightweight dispatch，减少 dtype/runtime dispatch 成本。

### P5：再考虑 GPU/Vulkan/CUDA 路径

今天测试目标是 CPU-only。CPU 后端先建立正确的 benchmark 和关键内核优化后，再回到 Vulkan/CUDA 会更稳。否则 GPU 路径快慢也难以解释。

## 下一步建议

1. 新增 `benchmark_decode.cpp`：只测 prefill/decode/sampling 三段。
2. 新增 `bench_linear_iq4nl.cpp`：单独压测 `IQ4_NL x BF16 -> BF16` 的 GEMV。
3. 实现 fused IQ4_NL GEMV 专用内核。
4. 用同一模型重新对比 llama.cpp 的 `llama-bench`。
5. 把指标记录成固定表格，后续每次优化都记录同一组 prompt/token 参数。

## 今日结论

Genllm 已经能在 CPU-only 下跑通 Qwen3-0.6B IQ4_NL 的完整推理，但当前性能距离 llama.cpp 还有明显差距。最值得投入的方向不是调度器，而是 CPU 量化 GEMV 内核、attention 标量路径、sampling 计时隔离和 benchmark 体系。优先把 `IQ4_NL linear` 从“反量化到 BF16 再计算”改成“融合反量化与 dot product”，这是最可能带来数量级提升的优化点。
