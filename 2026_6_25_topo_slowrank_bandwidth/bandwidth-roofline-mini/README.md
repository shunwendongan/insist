# bandwidth-roofline-mini

一个很小的 CUDA bandwidth / roofline 入门 demo。

目标：

- 用 `vector_add` 测一个典型 memory-bound kernel 的有效带宽。
- 手写 CUDA `vector_add`，同时输出 CUDA event 计时和 `cudaDeviceSynchronize()` 同步计时。
- 用 `reduction_stage1` 对比另一个低算术强度 kernel，并输出 read/write bytes、FLOPs、算术强度和 roofline 区域判断。
- 通过多组 `N` 和 block size sweep 观察小规模 launch overhead 与大规模 HBM / global memory bandwidth 限制。

目录结构：

```text
bandwidth-roofline-mini/
├── README.md
├── src/
│   ├── vector_add.cu
│   └── reduction.cu
├── scripts/
│   └── bench.sh
└── notes/
    └── result.md
```

---

## 1. 在 Google Colab 上运行

先在 Colab 菜单里选：

```text
Runtime -> Change runtime type -> GPU
```

然后确认 GPU 和 CUDA 编译器可用：

```bash
!nvidia-smi
!nvcc --version
```

### 上传项目

在本机把这个目录压缩成 zip：

```text
E:\ai ifra demo\bandwidth-roofline-mini\bandwidth-roofline-mini
```

上传到 Colab 左侧 Files 面板后，假设文件名是 `bandwidth-roofline-mini.zip`，运行：

```bash
!unzip -q /content/bandwidth-roofline-mini.zip -d /content
%cd /content/bandwidth-roofline-mini
```

如果你上传后解压出来多了一层目录，用 `!ls /content` 看一下目录名，再 `%cd` 到包含 `src/` 和 `scripts/` 的那一层。

### 默认 sweep

```bash
!bash scripts/bench.sh
```

默认会测试：

```text
N_VALUES=1048576,16777216,67108864,268435456
BLOCK_VALUES=128,256,512
WARMUP=5
REPEAT=20
```

运行结束后结果会写到：

```bash
!cat notes/result.md
```

### 跑单组参数

如果只想快速看一个配置：

```bash
!N=268435456 BLOCK=256 WARMUP=10 REPEAT=50 bash scripts/bench.sh
```

如果想自定义多组 sweep：

```bash
!N_VALUES=1048576,16777216,268435456 BLOCK_VALUES=128,256,512 WARMUP=10 REPEAT=50 bash scripts/bench.sh
```

### 带 roofline 峰值参数运行

如果你知道当前 GPU 的 FP32 峰值和显存带宽，可以传入：

```bash
!PEAK_FP32_GFLOPS=8140 PEAK_MEM_GBS=320 N=268435456 BLOCK=256 bash scripts/bench.sh
```

程序会计算：

```text
roofline_knee = peak_fp32_gflops / peak_mem_gbs
```

并输出：

```text
roofline_region=memory_bound
```

或：

```text
roofline_region=compute_bound
```

如果不传峰值参数，程序会对低算术强度 kernel 输出：

```text
roofline_region=memory_bound_low_ai_expected
```

### 手动编译运行

```bash
!mkdir -p build
!nvcc -O3 src/vector_add.cu -o build/vector_add
!nvcc -O3 src/reduction.cu -o build/reduction
!./build/vector_add --n 268435456 --warmup 10 --repeat 50 --block 256
!./build/reduction --n 268435456 --warmup 10 --repeat 50 --block 256
```

你原始要求中的命令也可以跑：

```bash
!mkdir -p build
!nvcc -O3 src/vector_add.cu -o build/vector_add && ./build/vector_add --n 268435456 --repeat 50
```

---

## 2. 输出字段说明

`vector_add` 输出类似：

```text
RESULT,kernel=vector_add,dtype=float32,N=268435456,warmup=10,repeat=50,read_bytes=2147483648,write_bytes=1073741824,bytes_per_iter=3221225472,flops_per_iter=268435456,arithmetic_intensity_flop_per_byte=0.083333,roofline_region=memory_bound_low_ai_expected,roofline_knee_flop_per_byte=0.000000,event_p50_ms=...,event_p95_ms=...,sync_p50_ms=...,sync_p95_ms=...,GBs_p50=...,GBs_at_p95_latency=...
```

字段解释：

| 字段 | 含义 |
|---|---|
| dtype | 数据类型，这个 demo 默认是 float32 |
| N | 向量元素个数 |
| warmup | 正式计时前预热次数 |
| repeat | 正式计时次数 |
| read_bytes | 每次 kernel 理论 global memory 读取字节数 |
| write_bytes | 每次 kernel 理论 global memory 写入字节数 |
| bytes_per_iter | 每次 kernel 理论读写字节数 |
| flops_per_iter | 每次 kernel 估算 FLOPs |
| arithmetic_intensity_flop_per_byte | 算术强度，单位 FLOP/Byte |
| roofline_region | roofline 区域判断 |
| roofline_knee_flop_per_byte | 设置峰值参数时的 roofline 拐点 |
| event_p50_ms | CUDA event 计时的中位数延迟 |
| event_p95_ms | CUDA event 计时的 95 分位延迟 |
| sync_p50_ms | host 侧 `cudaDeviceSynchronize()` 同步计时的中位数延迟 |
| sync_p95_ms | host 侧同步计时的 95 分位延迟 |
| GBs_p50 | 用 p50 延迟计算出来的有效带宽 |
| GBs_at_p95_latency | 用 p95 延迟计算出来的有效带宽，通常比 p50 对应带宽低 |

---

## 3. 为什么 vector add 主要受 HBM 带宽限制？

vector add 做的是：

```cpp
C[i] = A[i] + B[i];
```

对每个 float32 元素：

- 读 `A[i]`：4 bytes
- 读 `B[i]`：4 bytes
- 写 `C[i]`：4 bytes
- 做 1 次浮点加法：1 FLOP

所以每个元素大约：

```text
访存量 = 12 bytes
计算量 = 1 FLOP
算术强度 = 1 / 12 ≈ 0.083 FLOP/Byte
```

roofline 公式：

```text
性能上限 = min(峰值算力, 显存带宽 × 算术强度)
```

因为 `vector_add` 的算术强度非常低，GPU 的 FP32 计算单元根本吃不饱。瓶颈不是“加法算不过来”，而是“数据从 HBM / 显存搬不过来”。

换句话说：

```text
vector add 不是缺计算，是缺喂数据的速度。
```

所以优化重点通常是：

- global memory 访问合并，也就是 coalesced load/store
- 减少不必要访存
- 避免非连续访问
- 尽量让访问模式简单、连续、对齐
- 用更大的 N 避免 kernel launch overhead 影响测量

---

## 4. reduction 的字节数和 roofline 判断

`reduction_stage1` 每个 block 输出一个 partial sum。这个 demo 只统计 stage1 kernel，不做最终二次规约。

理论 global memory traffic：

```text
read_bytes = N * sizeof(float)
write_bytes = grid_size * sizeof(float)
bytes_per_iter = read_bytes + write_bytes
```

其中：

```text
grid_size = ceil(N / (block_size * 2))
```

FLOPs 近似按每消掉一个元素做一次加法：

```text
flops_per_iter ≈ N - grid_size
```

所以当 N 很大时：

```text
arithmetic_intensity ≈ 1 FLOP / 4 Bytes = 0.25 FLOP/Byte
```

它比 `vector_add` 的 0.083 FLOP/Byte 稍高，但仍然很低。通常 GPU 的 roofline 拐点远高于这个值，所以它也主要落在 memory-bound / bandwidth-bound 区域。

---

## 5. 常用实验命令

测 256M 个 float，约 1Gi 个输入/输出数据量级：

```bash
./build/vector_add --n 268435456 --warmup 10 --repeat 50
```

测不同 N：

```bash
./build/vector_add --n 1048576 --warmup 10 --repeat 50
./build/vector_add --n 16777216 --warmup 10 --repeat 50
./build/vector_add --n 268435456 --warmup 10 --repeat 50
```

测不同 block size：

```bash
./build/vector_add --n 268435456 --warmup 10 --repeat 50 --block 128
./build/vector_add --n 268435456 --warmup 10 --repeat 50 --block 256
./build/vector_add --n 268435456 --warmup 10 --repeat 50 --block 512
```

测 reduction：

```bash
./build/reduction --n 1048576 --warmup 10 --repeat 50 --block 256
./build/reduction --n 16777216 --warmup 10 --repeat 50 --block 256
./build/reduction --n 268435456 --warmup 10 --repeat 50 --block 256
```

---

## 6. 怎么看是否受 HBM 限制

重点看三件事：

1. `arithmetic_intensity_flop_per_byte` 很低，并且 `roofline_region` 是 `memory_bound` 或 `memory_bound_low_ai_expected`。
2. 大 N 下 `GBs_p50` 逐渐接近某个稳定上限，小 N 下则更容易被 launch overhead 和同步开销污染。
3. 对大 N 来说，改 `block=128/256/512` 后 GB/s 变化不大，说明瓶颈更像 global memory bandwidth，而不是 block size 本身。

`event_p50_ms` 更接近 kernel 本体时间；`sync_p50_ms` 包含 host 发起 kernel 后等待 GPU 完成的同步边界，通常会略大一点。小 N 时二者差异会更明显。

---

## 7. 注意事项

1. Colab 免费 GPU 型号不固定，T4 / L4 / A100 等机器测出来会差很多。
2. 第一次运行可能受 GPU boost、温度、后台任务影响，所以要看多次 repeat 的 p50 / p95。
3. `GB/s` 是按理论字节数估算的有效带宽，不一定等于硬件标称 HBM 峰值带宽。
4. 小 N 很容易被 kernel launch overhead、计时抖动影响，不适合判断显存带宽上限。
5. 这个 demo 的 roofline 是教学级估算，读写字节数是理论 global memory traffic，不等同于 Nsight Compute 里观测到的实际硬件事务字节数。
