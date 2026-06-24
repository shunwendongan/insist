# CPU-only inference benchmark: llama.cpp vs modified Nano vLLM

- Run id: `20260625_001813`
- Date: `2026-06-25 00:18:24 +0800`
- Model: `E:\model\Qwen3-0.6B-IQ4_NL.gguf`
- Prompt: `请用三句话解释CPU-only LLM推理中 batch size 和线程数如何影响吞吐与延迟。`
- Max tokens: `48`
- Threads: `4`
- Context size: `512`
- Batch size / ubatch size: `512` / `512`
- Warmup rounds: `1`
- Serial measured rounds per system: `5`
- Paired concurrent rounds: `5`
- CPU scope: `CPU-only; CUDA_VISIBLE_DEVICES cleared for Nano; n_gpu_layers=0/default for GGUF backends`

## Commands

```powershell
E:\model\llama.cpp\llama-server.exe --model E:\model\Qwen3-0.6B-IQ4_NL.gguf --host 127.0.0.1 --port 18180 --ctx-size 512 --batch-size 512 --ubatch-size 512 --threads 4 --parallel 1 --no-cache-prompt --no-webui --metrics
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m nanovllm.serve --model E:\model\Qwen3-0.6B-IQ4_NL.gguf --host 127.0.0.1 --port 18181 --model-backend gguf --max-model-len 512 --gguf-n-ctx 512 --gguf-n-batch 512 --gguf-n-ubatch 512 --gguf-n-threads 4
```

## Summary

| Phase | System | OK | Avg latency s | P50 latency s | P95 latency s | Avg completion tok/s | Aggregate tok/s | Avg completion tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paired_concurrent | llama.cpp | 5/5 | 0.6370 | 0.6228 | 0.6685 | 75.436 | 75.352 | 48.0 |
| paired_concurrent | nano_vllm_gguf | 5/5 | 0.6774 | 0.6790 | 0.6934 | 70.884 | 70.859 | 48.0 |
| serial | llama.cpp | 5/5 | 0.5091 | 0.5107 | 0.5229 | 94.327 | 94.288 | 48.0 |
| serial | nano_vllm_gguf | 5/5 | 0.5647 | 0.5635 | 0.5787 | 85.031 | 85.006 | 48.0 |
| warmup | llama.cpp | 1/1 | 0.4976 | 0.4976 | 0.4976 | 96.455 | 96.455 | 48.0 |
| warmup | nano_vllm_gguf | 1/1 | 0.6254 | 0.6254 | 0.6254 | 76.754 | 76.754 | 48.0 |

## Per-request Results

| Phase | Round | System | Latency s | Completion tokens | Completion tok/s | Text preview |
|---|---:|---|---:|---:|---:|---|
| warmup | 1 | llama.cpp | 0.4976 | 48 | 96.455 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| warmup | 1 | nano_vllm_gguf | 0.6254 | 48 | 76.754 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 1 | llama.cpp | 0.4963 | 48 | 96.720 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 1 | nano_vllm_gguf | 0.5635 | 48 | 85.184 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 2 | llama.cpp | 0.5143 | 48 | 93.322 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 2 | nano_vllm_gguf | 0.5540 | 48 | 86.650 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 3 | llama.cpp | 0.4990 | 48 | 96.183 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 3 | nano_vllm_gguf | 0.5577 | 48 | 86.075 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 4 | llama.cpp | 0.5250 | 48 | 91.422 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 4 | nano_vllm_gguf | 0.5817 | 48 | 82.511 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 5 | llama.cpp | 0.5107 | 48 | 93.991 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| serial | 5 | nano_vllm_gguf | 0.5665 | 48 | 84.734 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 1 | llama.cpp | 0.6228 | 48 | 77.075 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 1 | nano_vllm_gguf | 0.6852 | 48 | 70.057 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 2 | llama.cpp | 0.6216 | 48 | 77.218 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 2 | nano_vllm_gguf | 0.6686 | 48 | 71.786 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 3 | llama.cpp | 0.6503 | 48 | 73.813 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 3 | nano_vllm_gguf | 0.6790 | 48 | 70.697 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 4 | llama.cpp | 0.6173 | 48 | 77.758 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 4 | nano_vllm_gguf | 0.6588 | 48 | 72.863 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 5 | llama.cpp | 0.6731 | 48 | 71.315 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |
| paired_concurrent | 5 | nano_vllm_gguf | 0.6955 | 48 | 69.016 |   用中文解释，不用翻译，用专业术语。  例如：CPU-only LLM推理中，当 batch size 为 1 时，每个模型的推理时间是 1秒，而 batch size  |

## Bottleneck Analysis

- Serial CPU throughput winner: `llama.cpp` by about `1.11x` on aggregate completion tok/s.
- Both systems are executing GGUF on CPU. The comparison mainly measures wrapper/server overhead plus llama.cpp CPU kernel behavior, not CUDA native nano-vLLM scheduling.
- `llama.cpp` uses the native `llama-server.exe` path. Nano vLLM uses the new `model_backend=gguf` wrapper around `llama-cpp-python`, with a serial lock in the backend.
- Prompt caching was disabled for `llama.cpp` with `--no-cache-prompt`; Nano GGUF reports prefix cache unsupported. Repeated prompts therefore should not get prefix-cache acceleration.
- The paired-concurrent phase intentionally creates CPU contention. Treat it as a co-residency stress check, not as the clean standalone throughput ranking.
- Paired concurrent aggregate tok/s: llama.cpp `75.352`, Nano GGUF `70.859`.
- If either system shows lower throughput with similar output token counts, the likely bottleneck is CPU decode kernel/runtime overhead rather than model size or GPU memory. Larger `--threads` sweeps can identify the local CPU saturation point.

## Artifacts

- `results.json`: full machine-readable results and metrics.
- `results.csv`: per-request table.
- `llama_server.log`: llama.cpp server log.
- `nano_vllm_server.log`: Nano vLLM server log.