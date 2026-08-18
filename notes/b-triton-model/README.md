# Module B — Triton's model: blocked programs vs CUDA threads

Reference: Tillet, Kung, Cox — *"Triton: An Intermediate Language and Compiler for Tiled
Neural-Network Computations"*, MAPL 2019. The artifact is this repo's own matched pair:
the same fused mul-add-relu written both ways, benchmarked on the same silicon.

- [`kernels/elementwise.py`](../../kernels/elementwise.py) — one Triton program, ~15 lines
- [`cuda/elementwise.cu`](../../cuda/elementwise.cu) — **three** CUDA kernels: scalar,
  `float4`-vectorized, and a scalar tail to clean up after the vectorized one

## The core idea

CUDA's contract: you write a *scalar program per thread* and own the mapping from thread
IDs to data. Triton's contract (the MAPL paper's thesis): you write a program per *block
of data* — `tl.arange(0, BLOCK_SIZE)` names a tile, operations act on whole tiles — and
the mapping of tile elements onto threads, along with everything downstream of that
choice, belongs to the compiler.

That one inversion moves a whole class of decisions across the interface:

| decision | in `elementwise.py` | in `elementwise.cu` |
|---|---|---|
| element → thread mapping | compiler (layout: `sizePerThread`/warps) | me: index arithmetic + grid-stride loop |
| vectorized access | compiler derives `ld.global.v4` from layout + alignment | me: a second, hand-written `float4` kernel |
| ragged boundary | `mask=` → predication | me: a third kernel for the tail |
| launch config | `num_warps` hint, autotunable | me: `<<<grid, block>>>` per call site |
| specialization to runtime facts | JIT: one source, many binaries | me: I *am* the specializer — one source file per variant |

## What the measurements say

From the root README's table (same clocks, committed CSVs): at 2²⁶ elements Triton's
generated code lands **within 0.7% of the hand-written `float4` kernel** (97.0 vs 97.7
GB/s), while the hand-written *scalar* kernel drops to 85.9. The single decision that
separates the good hand kernel from the mediocre one — vectorized access — is precisely
the one Triton automates. The abstraction isn't costing performance; it is supplying the
optimization I'd otherwise owe by hand, times every kernel, times every dtype.

## The receipt that CUDA can't write

Module C holds two committed PTX files compiled from the *same* Triton source:

- [`c-llvm/elementwise_block1024_fp32.ptx`](../c-llvm/elementwise_block1024_fp32.ptx) —
  `ld.global.v4.b32` vector loads (benchmark sizes: JIT observed `n_elements % 16 == 0`)
- [`c-llvm/elementwise_block1024_fp32_ragged.ptx`](../c-llvm/elementwise_block1024_fp32_ragged.ptx)
  — 24 scalar predicated loads (test size n=4099: a 4-wide load could straddle the mask edge)

One source; the compiler chose per *runtime facts*. The C++ file expresses the same space
the only way a static language can — by me writing each point in it as a separate kernel.
That's the paper's argument made concrete on my own silicon.

## Where the model costs

Honesty section: the blocked model's scheduling freedom is per-kernel, not per-problem.
My softmax loses to torch at narrow rows because one program per row under-occupies the
machine — Inductor's `XBLOCK` row-batching (see [module E](../e-inductor/README.md)) is a
*tiling* decision above the kernel that Triton doesn't make for you. And matmul at 4096³
shows the compiler doesn't relieve you of blocking strategy either (74–82% of cuBLAS).
Triton moves the instruction-level decisions inside the compiler; the algorithm-level ones
are still yours.
