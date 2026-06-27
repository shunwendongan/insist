# Softmax Profile Card

## Verdict

- Correctness 对拍: PASS ✅
- Local mode: PyTorch CPU fallback
- GPU profiling fields are kept in the report, but marked N/A without CUDA/NCU.

## Environment

- Generated at: `2026-06-26 23:36:08`
- Python: `3.12.8`
- Platform: `Windows-11-10.0.26200-SP0`
- PyTorch: `2.12.1+cpu`
- Device: `cpu`
- Benchmark shape: `4096 x 1024`
- Benchmark dtype: `float32`
- Iterations: `3000`

## Correctness 对拍

Tolerance: `rtol=1e-05`, `atol=1e-06`, `equal_nan=True`

| Case | Shape | DType | Contiguous | Max Abs Error | Max Rel Error | Pass | Note |
|---|---:|---|---|---:|---:|---|---|
| `normal_2d` | `(128, 256)` | `float32` | `True` | `1.490e-08` | `3.512e-07` | ✅ | common 2D input |
| `single_row` | `(1, 1024)` | `float32` | `True` | `1.863e-09` | `2.989e-07` | ✅ | batch/row boundary |
| `single_col` | `(128, 1)` | `float32` | `True` | `0.000e+00` | `0.000e+00` | ✅ | softmax result should be 1 |
| `non_power_of_2_cols` | `(37, 1000)` | `float32` | `True` | `5.588e-09` | `4.718e-07` | ✅ | cols is not power of 2 |
| `large_positive_negative` | `(1, 4)` | `float32` | `True` | `0.000e+00` | `0.000e+00` | ✅ | numerical stability |
| `all_zeros` | `(32, 128)` | `float32` | `True` | `0.000e+00` | `0.000e+00` | ✅ | uniform output expected |
| `float64` | `(64, 257)` | `float64` | `True` | `4.163e-17` | `6.820e-16` | ✅ | dtype check |
| `non_contiguous` | `(64, 1024)` | `float32` | `False` | `3.725e-09` | `4.445e-07` | ✅ | strided tensor, non-contiguous |
| `nan_inf` | `(2, 3)` | `float32` | `True` | `0.000e+00` | `0.000e+00` | ✅ | explicit abnormal values |

## CPU Timing

`estimated_effective_GB/s` is a local approximation, not real GPU global-memory throughput.

| Impl | Avg Time / Iter | Estimated Effective Throughput |
|---|---:|---:|
| `reference_torch_softmax` | `0.4527 ms` | `185.321 GB/s` |
| `candidate_stable_softmax_cpu` | `2.3365 ms` | `35.903 GB/s` |

## GPU / NCU Metrics

| Metric | Local CPU Value | How to collect later on GPU |
|---|---:|---|
| Global memory throughput | N/A | Use Nsight Compute metric groups from `ncu --set full` |
| Warp stall | N/A | Inspect Warp State / Scheduler Stats in NCU |
| Registers/thread | N/A | Inspect Launch Statistics in NCU |

## Commands

CPU local:

```bash
python tests/test_softmax.py
python tests/test_softmax.py --rows 4096 --cols 1024 --iters 30
```

GPU later:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ncu --set full ./build/softmax_bench --rows 4096 --cols 1024
```

## Notes

- `reference.py` is the correctness baseline, not the performance baseline.
- `kernels/softmax_naive.cu` is intentionally simple and slow; it is for correctness intuition.
- `kernels/softmax_block.cu` shows the block-per-row reduction style that is closer to real CUDA softmax kernels.
- On CPU, this demo only validates logic and report format. Real `global memory throughput`, `warp stall`, and `registers/thread` require CUDA + NCU.
