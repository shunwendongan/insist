"""CPU-friendly softmax correctness + profiling demo.

Run:
    python tests/test_softmax.py

Optional:
    python tests/test_softmax.py --rows 4096 --cols 1024 --iters 30

This script does three things:
1. Uses reference.py as the correctness baseline.
2. Runs several edge/corner cases for 对拍.
3. Writes a profile card to profiles/softmax.md.

No CUDA GPU is required. GPU-only metrics such as warp stall and
registers/thread are reported as N/A in the generated markdown.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch

# Make project root importable when running from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference import reference_softmax, stable_softmax_cpu  # noqa: E402


@dataclass
class CaseResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    contiguous: bool
    max_abs_error: float
    max_rel_error: float
    passed: bool
    note: str = ""


@dataclass
class BenchResult:
    name: str
    avg_ms: float
    estimated_effective_gbs: float


def max_errors(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return max absolute and relative error, treating NaN matches as okay.

    For NaN positions where both outputs are NaN, error is treated as zero.
    This matters for explicit NaN/Inf abnormal cases.
    """
    out64 = out.detach().to(torch.float64)
    ref64 = ref.detach().to(torch.float64)

    both_nan = torch.isnan(out64) & torch.isnan(ref64)
    diff = torch.abs(out64 - ref64)
    diff = torch.where(both_nan, torch.zeros_like(diff), diff)

    denom = torch.clamp(torch.abs(ref64), min=1e-12)
    rel = diff / denom

    # If ref has zeros and out is exactly zero, rel is zero. If both are NaN,
    # the earlier mask has already set diff to zero.
    max_abs = float(torch.nan_to_num(diff, nan=0.0, posinf=float("inf")).max().item())
    max_rel = float(torch.nan_to_num(rel, nan=0.0, posinf=float("inf")).max().item())
    return max_abs, max_rel


def allclose_with_nan(out: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float) -> bool:
    return bool(torch.allclose(out, ref, rtol=rtol, atol=atol, equal_nan=True))


def build_cases() -> list[tuple[str, torch.Tensor, str]]:
    torch.manual_seed(0)
    cases: list[tuple[str, torch.Tensor, str]] = []

    cases.append(("normal_2d", torch.randn(128, 256, dtype=torch.float32), "common 2D input"))
    cases.append(("single_row", torch.randn(1, 1024, dtype=torch.float32), "batch/row boundary"))
    cases.append(("single_col", torch.randn(128, 1, dtype=torch.float32), "softmax result should be 1"))
    cases.append(("non_power_of_2_cols", torch.randn(37, 1000, dtype=torch.float32), "cols is not power of 2"))
    cases.append(("large_positive_negative", torch.tensor([[1000.0, 999.0, -1000.0, 0.0]], dtype=torch.float32), "numerical stability"))
    cases.append(("all_zeros", torch.zeros(32, 128, dtype=torch.float32), "uniform output expected"))
    cases.append(("float64", torch.randn(64, 257, dtype=torch.float64), "dtype check"))

    base = torch.randn(64, 2048, dtype=torch.float32)
    cases.append(("non_contiguous", base[:, ::2], "strided tensor, non-contiguous"))

    nan_inf = torch.tensor([[0.0, float("nan"), 1.0], [float("inf"), 1.0, -1.0]], dtype=torch.float32)
    cases.append(("nan_inf", nan_inf, "explicit abnormal values"))

    return cases


def run_correctness(rtol: float, atol: float) -> list[CaseResult]:
    results: list[CaseResult] = []
    for name, x, note in build_cases():
        ref = reference_softmax(x, dim=-1)
        out = stable_softmax_cpu(x, dim=-1)
        max_abs, max_rel = max_errors(out, ref)
        passed = allclose_with_nan(out, ref, rtol=rtol, atol=atol)
        results.append(
            CaseResult(
                name=name,
                shape=tuple(x.shape),
                dtype=str(x.dtype).replace("torch.", ""),
                contiguous=bool(x.is_contiguous()),
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                passed=passed,
                note=note,
            )
        )
    return results


def time_function(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        y = fn(x)
    # Touch result once to avoid overly clever laziness in other frameworks.
    _ = float(y.reshape(-1)[0].item())

    start = time.perf_counter()
    for _ in range(iters):
        y = fn(x)
    _ = float(y.reshape(-1)[0].item())
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def estimate_softmax_bytes(x: torch.Tensor) -> int:
    """Rough effective traffic estimate for a stable softmax-like pipeline.

    This is not hardware DRAM traffic. It is only a CPU-friendly approximation
    for comparing local runs before moving to NCU.

    Approximation:
    - read input for max reduction
    - read input again for exp
    - write temporary numerator
    - read temporary for sum/final normalize
    - write output
    """
    return int(x.numel() * x.element_size() * 5)


def run_benchmark(rows: int, cols: int, dtype: torch.dtype, iters: int, warmup: int) -> list[BenchResult]:
    torch.manual_seed(123)
    x = torch.randn(rows, cols, dtype=dtype)
    byte_estimate = estimate_softmax_bytes(x)

    bench_items: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("reference_torch_softmax", lambda t: reference_softmax(t, dim=-1)),
        ("candidate_stable_softmax_cpu", lambda t: stable_softmax_cpu(t, dim=-1)),
    ]

    results: list[BenchResult] = []
    for name, fn in bench_items:
        avg_ms = time_function(fn, x, iters=iters, warmup=warmup)
        seconds = avg_ms / 1000.0
        gbs = (byte_estimate / seconds) / 1e9 if seconds > 0 else math.inf
        results.append(BenchResult(name=name, avg_ms=avg_ms, estimated_effective_gbs=gbs))
    return results


def md_bool(v: bool) -> str:
    return "✅" if v else "❌"


def write_profile_card(
    path: Path,
    case_results: list[CaseResult],
    bench_results: list[BenchResult],
    rows: int,
    cols: int,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    iters: int,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_passed = all(r.passed for r in case_results)

    lines: list[str] = []
    lines.append("# Softmax Profile Card")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- Correctness 对拍: {'PASS ✅' if all_passed else 'FAIL ❌'}")
    lines.append("- Local mode: PyTorch CPU fallback")
    lines.append("- GPU profiling fields are kept in the report, but marked N/A without CUDA/NCU.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append(f"- PyTorch: `{torch.__version__}`")
    lines.append(f"- Device: `cpu`")
    lines.append(f"- Benchmark shape: `{rows} x {cols}`")
    lines.append(f"- Benchmark dtype: `{str(dtype).replace('torch.', '')}`")
    lines.append(f"- Iterations: `{iters}`")
    lines.append("")
    lines.append("## Correctness 对拍")
    lines.append("")
    lines.append(f"Tolerance: `rtol={rtol}`, `atol={atol}`, `equal_nan=True`")
    lines.append("")
    lines.append("| Case | Shape | DType | Contiguous | Max Abs Error | Max Rel Error | Pass | Note |")
    lines.append("|---|---:|---|---|---:|---:|---|---|")
    for r in case_results:
        lines.append(
            f"| `{r.name}` | `{r.shape}` | `{r.dtype}` | `{r.contiguous}` | "
            f"`{r.max_abs_error:.3e}` | `{r.max_rel_error:.3e}` | {md_bool(r.passed)} | {r.note} |"
        )
    lines.append("")
    lines.append("## CPU Timing")
    lines.append("")
    lines.append("`estimated_effective_GB/s` is a local approximation, not real GPU global-memory throughput.")
    lines.append("")
    lines.append("| Impl | Avg Time / Iter | Estimated Effective Throughput |")
    lines.append("|---|---:|---:|")
    for r in bench_results:
        lines.append(f"| `{r.name}` | `{r.avg_ms:.4f} ms` | `{r.estimated_effective_gbs:.3f} GB/s` |")
    lines.append("")
    lines.append("## GPU / NCU Metrics")
    lines.append("")
    lines.append("| Metric | Local CPU Value | How to collect later on GPU |")
    lines.append("|---|---:|---|")
    lines.append("| Global memory throughput | N/A | Use Nsight Compute metric groups from `ncu --set full` |")
    lines.append("| Warp stall | N/A | Inspect Warp State / Scheduler Stats in NCU |")
    lines.append("| Registers/thread | N/A | Inspect Launch Statistics in NCU |")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("CPU local:")
    lines.append("")
    lines.append("```bash")
    lines.append("python tests/test_softmax.py")
    lines.append("python tests/test_softmax.py --rows 4096 --cols 1024 --iters 30")
    lines.append("```")
    lines.append("")
    lines.append("GPU later:")
    lines.append("")
    lines.append("```bash")
    lines.append("cmake -S . -B build -DCMAKE_BUILD_TYPE=Release")
    lines.append("cmake --build build -j")
    lines.append("ncu --set full ./build/softmax_bench --rows 4096 --cols 1024")
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `reference.py` is the correctness baseline, not the performance baseline.")
    lines.append("- `kernels/softmax_naive.cu` is intentionally simple and slow; it is for correctness intuition.")
    lines.append("- `kernels/softmax_block.cu` shows the block-per-row reduction style that is closer to real CUDA softmax kernels.")
    lines.append("- On CPU, this demo only validates logic and report format. Real `global memory throughput`, `warp stall`, and `registers/thread` require CUDA + NCU.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp64", "float64", "double"}:
        return torch.float64
    raise ValueError(f"Unsupported CPU demo dtype: {name}. Use float32 or float64.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "fp32", "float64", "fp64", "double"])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--threads", type=int, default=0, help="Set torch CPU threads; 0 keeps PyTorch default.")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    dtype = parse_dtype(args.dtype)
    case_results = run_correctness(rtol=args.rtol, atol=args.atol)
    bench_results = run_benchmark(
        rows=args.rows,
        cols=args.cols,
        dtype=dtype,
        iters=args.iters,
        warmup=args.warmup,
    )

    report_path = ROOT / "profiles" / "softmax.md"
    write_profile_card(
        path=report_path,
        case_results=case_results,
        bench_results=bench_results,
        rows=args.rows,
        cols=args.cols,
        dtype=dtype,
        rtol=args.rtol,
        atol=args.atol,
        iters=args.iters,
    )

    failed = [r for r in case_results if not r.passed]
    print(f"correctness_cases={len(case_results)} failed={len(failed)} report={report_path}")
    for b in bench_results:
        print(f"{b.name}: avg_ms={b.avg_ms:.4f} estimated_effective_GB/s={b.estimated_effective_gbs:.3f}")

    if failed:
        print("Failed cases:")
        for r in failed:
            print(f"  - {r.name}: max_abs={r.max_abs_error:.3e}, max_rel={r.max_rel_error:.3e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
