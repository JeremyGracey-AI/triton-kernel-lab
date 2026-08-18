# Module C — LLVM foundations, read off this repo's own kernel

The artifacts are the fused elementwise kernel's four compile stages, captured from
Triton's cache after real runs on the Orin (fp32, BLOCK_SIZE=1024), committed unedited:

| file | stage |
|---|---|
| [`elementwise_block1024_fp32.ttir`](elementwise_block1024_fp32.ttir) | Triton IR (MLIR) |
| [`elementwise_block1024_fp32.ttgir`](elementwise_block1024_fp32.ttgir) | Triton GPU IR (MLIR + layouts) |
| [`elementwise_block1024_fp32.llir`](elementwise_block1024_fp32.llir) | **LLVM IR** — this module's subject |
| [`elementwise_block1024_fp32.ptx`](elementwise_block1024_fp32.ptx) | PTX from LLVM's NVPTX backend |
| [`elementwise_block1024_fp32_ragged.ptx`](elementwise_block1024_fp32_ragged.ptx) | same source, different specialization (below) |

## SSA, in the file

Every value in the `.llir` is defined exactly once and never mutated — that's static
single assignment, and it's what makes data-flow reasoning cheap for every pass
downstream:

```llvm
%8  = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()   ; my tt.get_program_id
%10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()     ; the thread's lane
```

This kernel is loop-free, so the file is one straight-line basic block and no phi nodes
appear. Where SSA gets interesting is a loop-carried value — the matmul K-loop
(`kernels/matmul.py`) carries its accumulator and advancing pointers around a back-edge,
which in LLVM IR becomes `phi` nodes merging "initial value" with "value from last
iteration." Straight-line SSA is bookkeeping; phis are where it earns its keep.

Things worth noticing in the same file:

- **Types everywhere**: `ptr addrspace(1)` is a *global-memory* pointer — address spaces
  are how a single IR serves a machine with several distinct memories.
- **Intrinsics vs instructions**: `@llvm.maxnum.f32` and the NVVM special-register reads
  are intrinsic calls — target features expressed inside target-independent IR.
- **Masked loads are inline asm**: Triton emits predicated PTX (`@$2 ld.global.b32 ...`)
  as inline-assembly rather than LLVM masked-load intrinsics — a deliberate end-run around
  the backend where the backend would do worse. Compilers are pragmatic layer cakes.
- **`!dbg` metadata** threads every instruction back to `elementwise.py` line numbers —
  the same location chain module D traced, and what profilers use to blame Python.

## What the pass pipeline did

Between the `.ttir` and the `.ptx` you can catch passes red-handed:

- **FMA contraction**: I wrote `a * b + c`; the PTX contains eight `fma.rn.f32` and no
  separate `mul`/`add` — fused by the optimizer under contraction rules (this is the
  `enable_fp_fusion` knob Inductor's metadata exposed in module E).
- **relu became `max.f32`** with the literal `0f00000000` — `tl.maximum` → `llvm.maxnum`
  → one instruction; no branch anywhere.
- **The backend boundary**: the PTX header (`.version 8.7`, `.target sm_87`) is where
  LLVM's NVPTX target stops. Register allocation happens *after* LLVM, inside `ptxas`
  (PTX is an infinite-register ISA) — the measured **34 registers/thread** in module A's
  worksheet exists nowhere in any of these files; it's a SASS-level fact.

## One attribute, two machine codes

The two committed PTX files come from the *identical* Python source. The whole difference
sits in the TTIR signatures:

```mlir
%n_elements: i32 {tt.divisibility = 16 : i32}   ; benchmark sizes — all multiples of 16
%n_elements: i32                                 ; test size n=4099
```

With divisibility known, a 4-wide load can never straddle the mask boundary, so the JIT
emits `ld.global.v4.b32` (128-bit, coalesced). Without it, the same loads compile to 24
scalar predicated `ld.global.b32`. One runtime fact, harvested by specialization, is the
difference between the vectorized code path and the fallback — the same information an
ahead-of-time compiler must either prove statically or forfeit. That trade — proof vs
specialization vs a slow generic path — is arguably *the* recurring decision in compiler
design, and here it is in two files you can diff.
