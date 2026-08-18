# pyright: reportArgumentType=false
# (Triton's stubs type tl.constexpr params as `constexpr`, so every launch
#  passing a plain int misfires this check.)
"""Fused elementwise: out = relu(a * b + c).

One program covers BLOCK_SIZE contiguous elements of the flattened tensors;
a 1-D grid covers the rest. The tail program is masked rather than padded.

The honest framing for the benchmark: this op is memory-bound. PyTorch eager
executes it as three kernels (mul, add, relu), each reading and writing global
memory, so the fused kernel's win comes from removing two full round trips of
memory traffic — not from any arithmetic cleverness. A lone `add` written in
Triton merely matches eager, because eager's single-kernel add is already at
bandwidth.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mul_add_relu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    out = tl.maximum(a * b + c, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_mul_add_relu(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape} vs {c.shape}")
    a, b, c = a.contiguous(), b.contiguous(), c.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_mul_add_relu_kernel[grid](a, b, c, out, n, BLOCK_SIZE=block_size)
    return out


def eager_mul_add_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Baseline: what the fused kernel is measured against (three eager kernels)."""
    return torch.relu(a * b + c)
