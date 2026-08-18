# triton-kernel-lab

Hand-written Triton GPU kernels with honest benchmarks, alongside a study track through the
compiler stack underneath them: LLVM → MLIR → Triton's lowering pipeline → TorchDynamo/Inductor.

Benchmarks run on hardware I own — an NVIDIA Jetson Orin Nano Super (Ampere, sm_87,
102 GB/s memory bandwidth) — with locked clocks and full environment metadata committed next
to every number. The interesting constraint of an 8 GB edge board is that it makes memory-traffic
accounting the whole game, which is the right instinct to train anyway.

## Kernels

| Kernel | What it exercises | Metric |
|---|---|---|
| `kernels/elementwise.py` — fused mul-add-relu | masking, coalescing, bandwidth accounting | GB/s vs ~102 GB/s device peak |
| `kernels/softmax.py` — fused row softmax + online variant | reductions, numerical stability (max-subtract), warp tuning, the online-softmax recurrence Flash-Attention builds on | GB/s vs eager `torch.softmax` |
| `kernels/matmul.py` — tiled matmul *(planned, M2)* | block tiling, fp32 accumulators, `@triton.autotune`, grouped ordering for L2 reuse | TFLOP/s and **% of cuBLAS** |

The framing throughout is deliberately honest. Fused elementwise beats eager because eager
launches three kernels, not because the arithmetic is clever — a lone Triton `add` merely
matches eager. And nobody hand-beats cuBLAS at matmul; the credible number is the percentage
of it you reach and an explanation of the gap.

## Methodology

- **Correctness before every timing.** `torch.testing.assert_close` against an fp32 reference
  with dtype-scaled tolerances, inside the bench loop itself — a number for a wrong kernel
  cannot be produced.
- **`triton.testing.do_bench`**: warmup, median + interquartile range, fp16 and fp32 sweeps.
- **Reproducibility on owned silicon**: MAXN power mode, `jetson_clocks` locked, cooldown
  pauses between sweep points, and torch/triton/CUDA versions + power mode recorded in every
  CSV row (`bench/results/<device>/`, committed). One command reproduces any table.
- The Jetson also runs resident services; `scripts/bench_guss.sh` pauses them, drops page
  caches, locks clocks, runs the sweep, and restores everything on exit — with every action
  logged to a committed session log.

## Running

```bash
# Correctness tests, no GPU needed (Triton CPU interpreter; CI runs exactly this)
TRITON_INTERPRET=1 pytest -q      # or: make test

# On a CUDA box: same suite against the real compiler
TRITON_INTERPRET=0 pytest -q      # or: make test-gpu

# New device? Validate it first
python scripts/jetson_smoke.py    # or: make smoke

# Full sweep (Jetson, wrapped in the pause/restore session script)
make bench
python -m bench.run --kernel softmax --dtype fp16 --sizes 4096x1024   # targeted
```

Interpreter-mode tests are a correctness gate only — sequential numpy execution, no
performance signal, no bf16.

## Results

First committed numbers land with M1 (Jetson bring-up). Every number in these tables will
trace to a CSV in `bench/results/` carrying the environment that produced it.

## Study track (`notes/`)

Public notes with one artifact each — see [`notes/README.md`](notes/README.md):

- **A. GPU execution model** — SIMT, memory hierarchy, occupancy
- **B. Triton's programming model** — blocked programs vs CUDA threads
- **C. LLVM foundations** — SSA, IR structure, pass pipelines
- **D. Triton's lowering pipeline** — my softmax annotated through TTIR → TTGIR → LLVM IR → PTX
- **E. TorchDynamo/Inductor** — bytecode capture, guards, graph breaks; what `torch.compile`
  does to my softmax, and how its generated Triton compares to mine

## Roadmap

- [x] **M0** — scaffold, elementwise + softmax kernels, interpreter-verified tests, bench harness, CI
- [ ] **M1** — Jetson bring-up, first committed GPU numbers, softmax lowering artifact (notes D seed)
- [ ] **M2** — tiled matmul + autotune (% of cuBLAS), raw CUDA C++ elementwise baseline,
  reading Inductor's generated Triton for softmax
- [ ] **M3** — study notes A–D complete
- [ ] **M4** — Dynamo/Inductor module, `torch.compile` steady-state baseline, perf plots

## Toolchain notes

torch and triton are deliberately unpinned in `pyproject.toml`: each platform installs its own
build (Jetson: NVIDIA's JetPack wheel index; CI: CPU wheels) and the versions that produced any
benchmark are recorded with it. CI runs lint (ruff), typecheck (pyright), and the interpreter-mode
suite on every push.

Jetson bring-up gotchas (JetPack 6 / L4T r36, learned the hard way): pin torch **by index, not
just version** — a bare resolve happily grabs PyPI's SBSA cu13 aarch64 wheel, which the Jetson
driver can't load (`--index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps torch==2.11.0`,
then deps from PyPI). And that wheel links `libcudss`, which JetPack doesn't ship —
`pip install nvidia-cudss-cu12`, then `source scripts/jetson_env.sh` puts the pip-installed
NVIDIA libs on `LD_LIBRARY_PATH` (the bench session script sources it automatically).
