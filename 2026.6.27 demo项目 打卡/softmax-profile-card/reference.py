"""Reference implementation for softmax correctness checking.

This file is intentionally simple: it uses PyTorch's trusted softmax as the
correctness baseline. It is not meant to be the fastest implementation.
"""

from __future__ import annotations

import torch


def reference_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Correctness baseline: PyTorch softmax."""
    return torch.softmax(x, dim=dim)


def stable_softmax_cpu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Candidate CPU implementation used by the local demo.

    It mirrors the numerically stable softmax formula:
        softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))

    In a real AI infra workflow, this function is where a CUDA/Triton kernel
    result would be compared against reference_softmax.
    """
    max_val = torch.max(x, dim=dim, keepdim=True).values
    z = x - max_val
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator
