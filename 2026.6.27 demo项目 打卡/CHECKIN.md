# 2026-06-27 AI Infra demo 项目打卡

## 来源

- 本地目录：`D:\Dev\Projects\ai ifra demo`
- 归档用途：作为 2026-06-27 今日学习成果，保存两个子项目的测试数据、验证脚本和性能分析报告。

## 子项目 1：topology / slow rank / bandwidth roofline

目录：`2026_6_25_topo_slowrank_bandwidth/`

已归档内容：

- `bandwidth-roofline-mini/README.md`
- `bandwidth-roofline-mini/notes/result.md`
- `bandwidth-roofline-mini/scripts/bench.sh`
- `bandwidth-roofline-mini/src/vector_add.cu`
- `bandwidth-roofline-mini/src/reduction.cu`
- `comm-topology-lab.zip`
- `comm-topology-lab-v2.zip`
- `nccl_slow_rank_analysis_flow (1).md`

学习重点：

- 用 `vector_add` 和 `reduction_stage1` 建立 bandwidth / roofline 直觉。
- 区分低算术强度 kernel 的 memory-bound 行为和 compute-bound 行为。
- 整理 NCCL slow rank / 通信拓扑分析流程。

## 子项目 2：softmax profile card

目录：`softmax-profile-card/`

已归档内容：

- `profiles/softmax.md`
- `tests/test_softmax.py`
- `reference.py`
- `kernels/softmax_naive.cu`
- `kernels/softmax_block.cu`
- `src/softmax_bench.cu`
- `CMakeLists.txt`
- `README.md`

本地报告摘要：

- Correctness 对拍：PASS
- Local mode：PyTorch CPU fallback
- Python：`3.12.8`
- PyTorch：`2.12.1+cpu`
- Benchmark shape：`4096 x 1024`
- Benchmark dtype：`float32`
- Iterations：`3000`
- `reference_torch_softmax`：`0.4527 ms/iter`，估算 `185.321 GB/s`
- `candidate_stable_softmax_cpu`：`2.3365 ms/iter`，估算 `35.903 GB/s`

说明：GPU / NCU 指标在本地 CPU 环境下标记为 N/A，后续需要在 CUDA + Nsight Compute 环境中采集真实 global memory throughput、warp stall、registers/thread 等指标。
