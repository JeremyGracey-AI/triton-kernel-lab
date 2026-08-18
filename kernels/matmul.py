# pyright: reportArgumentType=false, reportCallIssue=false
# (Triton's stubs type tl.constexpr params as `constexpr` and don't model
#  launch-time meta-parameters; both checks misfire on every kernel launch.)
"""Tiled matmul: C = A @ B with fp32 accumulation and autotuned block shapes.

Each program computes one BLOCK_M x BLOCK_N tile of C, marching along K in
BLOCK_K steps and accumulating with tl.dot into fp32 registers regardless of
input dtype. Program order is *grouped*: GROUP_M tile-rows are walked
column-major before moving on, so programs resident on the GPU at the same
time share B-tiles (and A-tiles) in L2 instead of streaming the whole matrix
per tile-row. `@triton.autotune` picks the block shape / stages / warps per
(M, N, K) from a small grid sized for an 8 GB Ampere part.

Load masking uses the wrap trick from the Triton tutorial: row/col offsets
are taken modulo M/N so out-of-range programs read valid (wrong) data with
no mask cost in the hot loop, and the store mask discards those lanes. Only
the K-tail needs a real load mask.

Precision note, stated rather than hidden: on Ampere-class hardware tl.dot
uses TF32 for fp32 inputs (10-bit mantissa, ~1e-3 relative error). cuBLAS
via torch.matmul defaults to IEEE fp32, so the benchmark enables TF32 for
the baseline to compare like for like — see bench/run.py.
"""

import os

import torch
import triton
import triton.language as tl

# The autotuner would time every config through the numpy interpreter when
# TRITON_INTERPRET=1; a single config keeps interpreter tests fast (with one
# config the autotuner skips benchmarking entirely).
_INTERPRETING = os.environ.get("TRITON_INTERPRET") == "1"


def _configs():
    if _INTERPRETING:
        return [
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 4},
                num_stages=1,
                num_warps=2,
            )
        ]
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering: walk GROUP_M tile-rows column-by-column for L2 reuse.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D tensors, got {a.ndim}-D @ {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dims differ: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} vs {b.dtype}")
    a, b = a.contiguous(), b.contiguous()
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c
