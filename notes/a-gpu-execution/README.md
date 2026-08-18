# Module A — GPU execution model: an occupancy & bandwidth worksheet

Worked on this repo's own fused elementwise kernel (`kernels/elementwise.py`,
BLOCK_SIZE=1024, fp32) on the machine that benchmarked it. No hypotheticals: device facts
come from the CUDA runtime (captured in `notes/e-inductor/`'s DeviceProperties dump),
kernel facts from `cuobjdump --dump-resource-usage` on the compiled cubin, and the
measured numbers from committed CSVs.

## The facts

| | value | source |
|---|---|---|
| SMs | 8 | DeviceProperties (`multi_processor_count`) |
| Max threads / SM | 1536 (48 warps) | DeviceProperties |
| Registers / SM | 65,536 | DeviceProperties |
| Peak DRAM bandwidth | ~102 GB/s | Orin Nano Super spec |
| Kernel block | 1024 threads = 32 warps | wrapper default |
| Registers / thread | **34** | `cuobjdump: REG:34 STACK:0 SHARED:0 LOCAL:0` |
| Shared memory | 0 | same |

## Occupancy

Blocks resident per SM is the minimum over each resource budget:

- **Threads:** ⌊1536 / 1024⌋ = **1 block** ← the binding limit
- **Registers:** 34 × 1024 = 34,816 of 65,536 → would allow 1 block anyway (and only just —
  2 blocks would need 69,632; allocation granularity rounds 34 up, making it worse)
- **Shared memory:** unused, no limit

So one 32-warp block resides per SM: **32 / 48 warps = 67% occupancy**.

Counterfactual worth doing on paper: BLOCK_SIZE=512 (16 warps) → threads allow 3 blocks,
registers 34×512 = 17,408 allow 3 → 48/48 warps = **100% occupancy**.

## Why 67% was still enough

The committed result: this kernel moves **97.0 GB/s of a ~102 GB/s peak (95%)** at 2²⁶
elements. Occupancy exists to hide latency; a streaming kernel with 32 warps per SM issuing
independent loads already keeps the memory system saturated, so the extra 16 warps have
nothing left to earn. Occupancy is a *budget*, not a goal — chase it only when the measured
limiter is latency, and here the limiter is DRAM:

- Arithmetic intensity: 2 FLOPs (mul, add — the relu compare is folded free) per 16 bytes
  (three fp32 loads + one store) = **0.125 FLOP/byte**.
- At 102 GB/s that demands 12.75 GFLOP/s of a machine whose matmul sustains multiple
  TFLOP/s (committed: 4.9 TFLOP/s fp32) — compute outruns memory here by ~300×.
  This kernel lives at the far-left of the roofline; nothing about the SMs matters except
  that there are enough warps in flight.

Grid shape at 2²⁶ elements: 65,536 blocks over 8 SMs = 8,192 waves — launch-tail effects
are noise. At the *small* end of the sweep (2¹⁶: 64 blocks = 8 waves) they are not, which
is exactly where the measured GB/s drops to 44.6 (see `bench/results/orin/`).

## Memory-hierarchy footnote

This op streams every byte exactly once — L2 cannot help, which is why the kernel holds
95% of peak at any large size. Contrast the matmul results in the root README: same
machine, but a kernel *with* reuse falls from parity to 74% of cuBLAS exactly when the
working set outgrows L2. Which hierarchy level a kernel lives in is the first question;
occupancy arithmetic is the second.
