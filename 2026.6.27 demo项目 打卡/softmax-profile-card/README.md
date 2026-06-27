# softmax-profile-card

一个 AI infra 入门 demo，用 `reference.py` 做 softmax 正确性 baseline，然后生成 profile card。

你现在本地没有显卡，所以默认路径是 **PyTorch CPU fallback**：

```bash
python tests/test_softmax.py
```

运行后会自动更新：

```text
profiles/softmax.md
```

里面会记录：

- 对拍误差
- 耗时
- CPU 估算 effective throughput
- global memory throughput / warp stall / registers/thread 的 GPU 占位说明

## 目录结构

```text
softmax-profile-card/
├── reference.py
├── tests/
│   └── test_softmax.py
├── kernels/
│   ├── softmax_naive.cu
│   └── softmax_block.cu
├── src/
│   └── softmax_bench.cu
├── profiles/
│   └── softmax.md
├── CMakeLists.txt
└── README.md
```

## CPU 本地运行

建议先进入工程目录：

```bash
cd softmax-profile-card
python tests/test_softmax.py
```

指定 shape：

```bash
python tests/test_softmax.py --rows 4096 --cols 1024 --iters 30
```

可选参数：

```bash
python tests/test_softmax.py --rows 4096 --cols 1024 --dtype float32 --iters 30 --threads 8
```

## 后续有 CUDA 显卡后

构建 CUDA benchmark：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

使用 Nsight Compute：

```bash
ncu --set full ./build/softmax_bench --rows 4096 --cols 1024
```

你原来的目标命令可以理解成：

```bash
python tests/test_softmax.py && ncu --set full ./build/softmax_bench --rows 4096 --cols 1024
```

在没有显卡的机器上，只运行前半段即可。

## 这个 demo 里的 baseline

`reference.py` 里面的：

```python
reference_softmax(x)
```

调用的是：

```python
torch.softmax(x, dim=-1)
```

它是 correctness baseline，意思是：你的候选实现必须和它对拍。

CPU demo 里的候选实现是：

```python
stable_softmax_cpu(x)
```

它模拟 CUDA kernel 的稳定 softmax 逻辑：

```text
max -> exp(x - max) -> sum -> normalize
```

## 为什么 GPU 指标是 N/A

这些指标需要真实 CUDA kernel + Nsight Compute：

- global memory throughput
- warp stall
- registers/thread

CPU 上没有 warp，也没有 CUDA register/thread 这个概念，所以 demo 不会假造这些指标。报告里会保留表格位置，等你以后在 GPU 环境下填充。
