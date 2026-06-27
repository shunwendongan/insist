#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p build notes

echo "== GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found. Did you enable GPU runtime?"
fi

echo
echo "== NVCC =="
nvcc --version

echo
echo "== Build =="
nvcc -O3 src/vector_add.cu -o build/vector_add
nvcc -O3 src/reduction.cu -o build/reduction

WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"

if [[ -n "${N:-}" ]]; then
  N_VALUES="$N"
else
  N_VALUES="${N_VALUES:-1048576,16777216,67108864,268435456}"
fi

if [[ -n "${BLOCK:-}" ]]; then
  BLOCK_VALUES="$BLOCK"
else
  BLOCK_VALUES="${BLOCK_VALUES:-128,256,512}"
fi

IFS=',' read -r -a N_LIST <<< "$N_VALUES"
IFS=',' read -r -a BLOCK_LIST <<< "$BLOCK_VALUES"

ROOFLINE_ARGS=()
if [[ -n "${PEAK_FP32_GFLOPS:-}" ]]; then
  ROOFLINE_ARGS+=(--peak-fp32-gflops "$PEAK_FP32_GFLOPS")
fi
if [[ -n "${PEAK_MEM_GBS:-}" ]]; then
  ROOFLINE_ARGS+=(--peak-mem-gbs "$PEAK_MEM_GBS")
fi

RAW_OUT="$(mktemp)"

echo
echo "== Sweep config =="
echo "N_VALUES=$N_VALUES"
echo "BLOCK_VALUES=$BLOCK_VALUES"
echo "WARMUP=$WARMUP"
echo "REPEAT=$REPEAT"
if [[ ${#ROOFLINE_ARGS[@]} -gt 0 ]]; then
  echo "ROOFLINE_ARGS=${ROOFLINE_ARGS[*]}"
else
  echo "ROOFLINE_ARGS=not set; region uses low-arithmetic-intensity heuristic"
fi

for current_n in "${N_LIST[@]}"; do
  for current_block in "${BLOCK_LIST[@]}"; do
    echo
    echo "== Run vector_add: N=$current_n block=$current_block =="
    ./build/vector_add \
      --n "$current_n" \
      --warmup "$WARMUP" \
      --repeat "$REPEAT" \
      --block "$current_block" \
      "${ROOFLINE_ARGS[@]}" | tee -a "$RAW_OUT"

    echo
    echo "== Run reduction: N=$current_n block=$current_block =="
    ./build/reduction \
      --n "$current_n" \
      --warmup "$WARMUP" \
      --repeat "$REPEAT" \
      --block "$current_block" \
      "${ROOFLINE_ARGS[@]}" | tee -a "$RAW_OUT"
  done
done

{
  echo "# Result"
  echo
  echo "Generated at: $(date)"
  echo
  echo "## Environment"
  echo
  echo '```text'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
  echo
  nvcc --version
  echo '```'
  echo
  echo "## Config"
  echo
  echo "| item | value |"
  echo "|---|---:|"
  echo "| dtype | float32 |"
  echo "| N_VALUES | $N_VALUES |"
  echo "| warmup | $WARMUP |"
  echo "| repeat | $REPEAT |"
  echo "| BLOCK_VALUES | $BLOCK_VALUES |"
  echo "| PEAK_FP32_GFLOPS | ${PEAK_FP32_GFLOPS:-not set} |"
  echo "| PEAK_MEM_GBS | ${PEAK_MEM_GBS:-not set} |"
  echo
  echo "## Raw output"
  echo
  echo '```text'
  cat "$RAW_OUT"
  echo '```'
  echo
  echo "## Interpretation"
  echo
  echo "### vector_add 为什么主要受 HBM / 显存带宽限制？"
  echo
  echo 'vector_add 的核心计算是 `C[i] = A[i] + B[i]`。对每个 float32 元素，它读 A 4 bytes、读 B 4 bytes、写 C 4 bytes，总访存约 12 bytes，只做 1 次浮点加法。'
  echo
  echo '所以算术强度约为：'
  echo
  echo '```text'
  echo '1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte'
  echo '```'
  echo
  echo '按照 roofline 模型：'
  echo
  echo '```text'
  echo 'performance <= min(peak_flops, memory_bandwidth * arithmetic_intensity)'
  echo '```'
  echo
  echo '因为算术强度太低，vector_add 通常落在 roofline 左侧斜坡区域，也就是 memory-bound / bandwidth-bound。真正限制它的不是 FP32 加法吞吐，而是 HBM / 显存把 A、B、C 数据搬进搬出的速度。'
  echo
  echo '### reduction_stage1 的读写字节数和 roofline 区域'
  echo
  echo '本 demo 的 reduction 只统计 stage1 kernel 的 global memory 传输：读取 `N * sizeof(float)`，写出 `grid_size * sizeof(float)` 个 partial sum。shared memory 内部读写不算 HBM/global memory traffic。'
  echo
  echo '近似 FLOPs 按“每消掉一个元素做一次加法”估算，因此 `flops_per_iter ≈ N - grid_size`。算术强度大约接近 `1 FLOP / 4 Bytes = 0.25 FLOP/Byte`，仍然非常低，通常也在 memory-bound 区域。'
  echo
  echo '如果设置了 `PEAK_FP32_GFLOPS` 和 `PEAK_MEM_GBS`，程序会用 `knee = peak_fp32_gflops / peak_mem_gbs` 计算 roofline 拐点，并输出 `roofline_region=memory_bound` 或 `compute_bound`。没有设置时，输出使用低算术强度启发式：`memory_bound_low_ai_expected`。'
  echo
  echo '### 如何观察是否受 HBM 限制'
  echo
  echo '对比不同 N：小 N 更容易被 kernel launch overhead 和同步开销影响；随着 N 变大，如果 `GBs_p50` 逐渐接近平台稳定上限，且 `event_p50_ms` 近似随字节数线性增长，就更像 HBM/global memory bandwidth 受限。'
  echo
  echo '对比不同 block size：如果 block 从 128 到 512 的变化对大 N 的 GB/s 影响不大，而 N 才是主要影响因素，也说明瓶颈更偏显存带宽，而不是单纯 block 配置。'
} > notes/result.md

rm -f "$RAW_OUT"

echo
echo "Result written to notes/result.md"
