# Module D artifact — my softmax, lowered stage by stage

One real kernel from this repo — `kernels/softmax.py::_softmax_kernel`, the single-block
variant, specialized to **BLOCK_N=4096, fp32, num_warps=8** — captured from Triton's compile
cache (`~/.triton/cache`) after the first benchmark run on the Jetson Orin Nano (sm_87,
Triton 3.7.1). The four dumps are committed beside this file, unedited:

| Stage | File | Lines | What this stage decides |
|---|---|---:|---|
| Triton IR | [`softmax_block4096_fp32.ttir`](softmax_block4096_fp32.ttir) | 80 | *what* to compute, on tensors — hardware-free |
| Triton GPU IR | [`softmax_block4096_fp32.ttgir`](softmax_block4096_fp32.ttgir) | 81 | *where* each element lives: the thread/warp layout |
| LLVM IR | [`softmax_block4096_fp32.llir`](softmax_block4096_fp32.llir) | 351 | tensors dissolve into per-thread scalar SSA |
| PTX | [`softmax_block4096_fp32.ptx`](softmax_block4096_fp32.ptx) | 579 | the machine story: vector loads, shuffles, shared memory |

The Python source is ~15 lines of `tl.*` calls. Watching what each stage adds is the clearest
way I've found to see what the compiler earns on my behalf.

## TTIR — the program, hardware-free

Every line of my Python survives, renamed into SSA. `tl.arange` is `tt.make_range`, the
broadcast of `n_cols` is `tt.splat`, and the masked load carries my `other=-inf` as a fused
third operand — note `0xFF800000`, the bit pattern of fp32 −inf:

```mlir
%x = arith.constant dense<0xFF800000> : tensor<4096xf32>
%x_5 = tt.load %x_4, %mask_0, %x : tensor<4096x!tt.ptr<f32>>
```

`tl.max` is not a special instruction — it's `tt.reduce`, an op carrying a *region* with the
combinator inside (here `arith.maxnumf`; the sum reduction is the same shape with `arith.addf`).
The IR is still about a 4096-wide tensor; no thread exists yet.

Two details worth noticing: the argument attributes `tt.divisibility = 16` are *specialization*
— the JIT observed my actual pointers and strides were 16-divisible and compiled this variant
under that assumption (it matters below). And the `#loc` table at the bottom maps every op back
to `softmax.py` line numbers — this is how profilers attribute SASS back to Python.

## TTGIR — one new fact: the layout

Diff TTIR against TTGIR and the ops are **identical**. What's new is an attribute:

```mlir
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-warps" = 8, ttg.target = "cuda:87", ...}
```

Every tensor type gains `, #blocked`. That single annotation is the mapping decision:
8 warps × 32 lanes × 4 contiguous elements per thread = 1024 slots per pass, so each thread
owns 16 of the row's 4096 elements, in four 4-wide contiguous chunks. My `num_warps=8`
heuristic from the Python wrapper landed here as `ttg.num-warps`, and the target is pinned
(`cuda:87` — this artifact is Orin-specific). The math didn't change; its *distribution* did.

## LLVM IR — tensors dissolve

No tensor types survive. Each thread is now a straight-line scalar program over its 16
registers, indexed via NVVM intrinsics:

```llvm
%8 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()   ; my tt.get_program_id
%9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()     ; which lane am I
%69 = tail call float @llvm.maxnum.f32(float %38, float %39)
%70 = tail call float @llvm.maxnum.f32(float %69, float %40)  ; ... 15 in a chain
```

The 15-deep `maxnum` chain is the first level of the reduction: each thread folds its own 16
elements before any communication happens.

## PTX — where the layout pays rent

Four things in the final ISA trace directly back to decisions above:

**1. The loads are 128-bit vectors** — `sizePerThread=[4]` plus the `divisibility=16` hint
means each thread can pull its 4-wide chunk in one coalesced transaction, predicated on my mask
(`@%p1` — masking compiles to predication, not branches):

```ptx
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
```

**2. The reduction is three-tier.** Per-thread register chain (`max.f32` ×15), then an
intra-warp butterfly over lane-shuffles — no memory touched:

```ptx
shfl.sync.bfly.b32  %r70, %r69, 16, 31, -1;
max.f32             %r71, %r69, %r70;
shfl.sync.bfly.b32  %r72, %r71, 8, 31, -1;   // then 4, 2, 1
```

then, because 8 warps must agree on one row-max, a shared-memory handoff — this is why
`global_smem` exists at all in a kernel I wrote with no mention of shared memory:

```ptx
@%p5 st.shared.b32 [ %r18 + 0 ], %r19;
bar.sync 0;
@%p6 ld.shared.b32 %r20, [ %r21 + 0 ];
```

**3. `exp` is two instructions.** `math.exp` lowers to multiply-by-log₂e then the SFU's
base-2 exponential — `0f3FB8AA3B` is 1.4426950 = log₂e:

```ptx
mul.f32        %r106, %r90, 0f3FB8AA3B;
ex2.approx.f32 %r107, %r106;
```

**4. The divide stays a real divide** (`div.full.f32`, IEEE-rounded full-throughput variant) —
Triton does not silently rewrite `x / sum` into `x * (1/sum)`; that accuracy/speed trade would
be mine to make explicitly.

## What the compiler earned

I wrote *"take the max of this 4096-wide tensor."* The compiler chose: how 4096 elements map
onto 256 threads, 128-bit vectorized access, a register/shuffle/shared three-level reduction
tree, predication for the ragged edge, and the SFU trick for `exp`. None of that is visible —
or spellable — in the source language. That's the case for blocked-program abstractions, and
the cost is also visible: every one of those choices is now frozen per (constexpr, dtype,
divisibility) specialization, which is exactly why Triton caches one directory per variant.

## Reproducing

```bash
# on the CUDA box, after any run that compiled the kernel:
ls ~/.triton/cache/*/_softmax_kernel.{ttir,ttgir,llir,ptx}
# this variant: the dir whose ttir contains tensor<4096xf32> and whose ttgir has num-warps = 8
```
