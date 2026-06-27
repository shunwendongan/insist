# Result

这里是 benchmark 结果记录文件。

运行：

```bash
bash scripts/bench.sh
```

脚本会自动覆盖并生成本文件内容。

---

## 手动记录模板

| kernel | dtype | N | block | warmup | repeat | read bytes | write bytes | AI FLOP/Byte | roofline region | event p50 ms | sync p50 ms | GB/s p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| vector_add | float32 | 268435456 | 256 | 10 | 50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| reduction_stage1 | float32 | 268435456 | 256 | 10 | 50 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

---

## vector_add 的 roofline 解释

`vector_add`：

```cpp
C[i] = A[i] + B[i];
```

每个 float32 元素：

```text
read A: 4 bytes
read B: 4 bytes
write C: 4 bytes
compute: 1 FLOP
```

所以：

```text
bytes per element = 12 bytes
FLOPs per element = 1 FLOP
arithmetic intensity = 1 / 12 ≈ 0.083 FLOP/Byte
```

这个算术强度很低。根据 roofline：

```text
performance <= min(peak_flops, memory_bandwidth * arithmetic_intensity)
```

当算术强度很低时，`memory_bandwidth * arithmetic_intensity` 这一项远低于 `peak_flops`，所以瓶颈主要是 HBM / 显存带宽。

也就是说，vector add 不是计算不够快，而是数据搬运不够快。

---

## reduction_stage1 的 roofline 解释

这个 demo 的 reduction 只统计 stage1 kernel：

```text
read_bytes = N * sizeof(float)
write_bytes = grid_size * sizeof(float)
grid_size = ceil(N / (block_size * 2))
```

FLOPs 近似按每消掉一个元素做一次加法：

```text
flops_per_iter ≈ N - grid_size
```

当 N 足够大时，算术强度接近：

```text
1 FLOP / 4 Bytes = 0.25 FLOP/Byte
```

它仍然是低算术强度 kernel，通常也会落在 memory-bound / bandwidth-bound 区域。

脚本会覆盖并生成包含多组 `N` 和 `block` 的真实结果：

```bash
bash scripts/bench.sh
```
