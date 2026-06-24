# 2026-06-25 打卡记录

## 内容概览

今天的打卡内容分为两部分：

1. `cpu_only_inference/`
   - `llama.cpp` 与魔改 `Nano vLLM` 的 CPU-only GGUF 推理对比实验。
   - 包含中文总结报告、可复现实验脚本、原始 `json/csv` 结果和服务日志。
2. `practical_kernel/`
   - 今日学习记录中的 3 个 CUDA/FlashAttention 相关源码文件：
   - `flash_attention_simple_stand.cu`
   - `tiled_flash_att_mine.cu`
   - `tiled_flash_attention_stand_hpc.cu`

## CPU-only 实验结论

- 测试模型：`E:\model\Qwen3-0.6B-IQ4_NL.gguf`
- 限定范围：只测本机 `CPU-only`
- 固定条件：相同提示词、`threads=4`、`ctx=512`、`batch=512`、`ubatch=512`、`max_tokens=48`
- 顺序请求下：
  - `llama.cpp` 约 `94.288 tok/s`
  - `Nano vLLM GGUF backend` 约 `85.006 tok/s`
- 成对并发下：
  - `llama.cpp` 约 `75.352 tok/s`
  - `Nano vLLM GGUF backend` 约 `70.859 tok/s`

结论：当前这套魔改 Nano vLLM 已经具备可用的本地 GGUF CPU 推理能力，但在这次服务级 CPU-only 对比里，原生 `llama.cpp` 仍然更快，优势大约在 `1.06x` 到 `1.11x` 之间。主要差异更可能来自 Python/FastAPI/adapter 层与后端串行封装开销，而不是模型本身。

## 来源说明

- CPU-only 对比实验来源：
  - 本地报告：`C:\Users\Administrator\Desktop\CPU-only推理性能对比报告_20260625.md`
  - 原始实验目录：`C:\Users\Administrator\Desktop\cpu_only_inference_benchmark_20260625`
- 魔改 Nano vLLM 工作区快照：
  - 仓库路径：`C:\Users\Administrator\Desktop\nano_vllm_note - 副本`
  - 基线提交：`bc3c1b547752360fcd6dccbc4ee0bf47d4454af0`
  - 注意：实验是在带有未提交本地改动的工作区上完成的，因此这里记录的是本地快照结果，不是某个干净标签版本。
- 学习文件来源：
  - `G:\我的云端硬盘\随机学习笔记总结成品\practical_kernel`
