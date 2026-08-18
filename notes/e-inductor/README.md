# Module E seed — what `torch.compile` does to my softmax

Captured on the Jetson (torch 2.11.0, Inductor targeting sm_87) with
`TORCH_LOGS=output_code python scripts/dump_inductor_softmax.py` over fp16 `torch.softmax`
on a 64×1024 tensor. Raw dump committed unedited as
[`inductor_softmax_fp16_raw.txt`](inductor_softmax_fp16_raw.txt); this note is the read.

## The pipeline before any Triton exists

The dump opens with the FX graph fragment Dynamo/AOT handed to Inductor. Two things stand out:

```
%prepare_softmax_online_default = call_function[target=torch.ops.prims.prepare_softmax_online.default]
```

PyTorch decomposes softmax into a *named* online-softmax primitive (max and rescaled sum in
one pass — the same recurrence my `softmax_online` kernel implements by hand) plus `sub`,
`exp`, `div`, and dtype casts. The algorithm choice I made manually is encoded upstream of
codegen as an ATen/prims decomposition — Inductor never "discovers" the max-subtract trick;
it inherits it.

And the guard machinery is visible at the call boundary:

```python
assert_size_stride(arg0_1, (64, 1024), (1024, 1))
```

The compiled artifact is specialized to this exact shape and stride — a different shape
re-triggers compilation. My hand kernel handles any (rows, cols) with one compile per
`BLOCK_N` power-of-two instead.

## The generated kernel, against mine

The kernel name is the fusion receipt:
`triton_per_fused__softmax_exp_prepare_softmax_online_sub_0` — one kernel, and the metadata
confirms `num_load: 1, num_store: 1, num_reduction: 4`. Same single-pass structure as my
single-block variant. The differences:

| | mine (`kernels/softmax.py`) | Inductor's |
|---|---|---|
| Rows per program | 1 (`tl.program_id` = row) | `XBLOCK` rows — a 2-D `[XBLOCK, R0_BLOCK]` tile |
| Row handling | `BLOCK_N = next_power_of_2(n_cols)`, masked tail | `R0_BLOCK = 1024` baked in (shape-specialized, no mask needed) |
| Reduction | `tl.max` / `tl.sum` on a 1-D tensor | `triton_helpers.max2` / `tl.sum` on axis 1 of the tile |
| `exp` | computed once, reused for numerator and sum | **computed twice** (`tmp9` for the sum, `tmp15` for the quotient — same value, recomputed) |
| Math dtype | fp32 always (explicit `.to(tl.float32)`) | same — loads fp16, upcasts immediately |
| Tuning | my `num_warps` heuristic | `@triton_heuristics.persistent_reduction` with `size_hints` + device properties (8 SMs, cc 87) baked into the metadata |

The `XBLOCK` batching answers a question my own benchmarks raised: torch beat my kernel at
**narrow rows** (4096×256), and here is the mechanism — Inductor amortizes launch and
occupancy cost by giving each program several rows, where my kernel strands an entire
program on 256 columns. That is a config choice, not deeper magic, and it is exactly what
I would steal for an `XBLOCK`-style variant.

The double `exp` cuts the other way: Inductor recomputes `exp(x - max)` for the output
instead of reusing the tensor it just summed — a recompute-over-registers trade a hand
kernel doesn't have to make. On a memory-bound kernel it's nearly free; it is still the
kind of thing you only see by reading the code the compiler wrote.

`libdevice.exp` vs my `tl.exp` is a distinction without a difference: both reach the same
`ex2.approx` + log₂e multiply in PTX (see [module D](../d-lowering/README.md)).

## Takeaway

`torch.compile`'s softmax and mine agree on every load-bearing decision — one fused pass,
fp32 accumulation, max-subtract stability, whole-row-resident reduction — because those are
forced by the problem. Where we differ (row batching, shape specialization, recompute vs
reuse) is the actual design space, and it's legible only at this layer. The steady-state
`torch.compile` timing baseline lands with module E proper (M4).
