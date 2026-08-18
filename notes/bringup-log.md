# Bring-up log — every failure, root cause, and fix

Real failures from bringing this repo up on a Jetson Orin Nano (JetPack 6 / L4T r36), in
order. Each one cost time; each one taught something a clean run wouldn't have. Session logs
under `bench/results/` are the unedited receipts.

## 1. A test suite that can't run everywhere must fail loudly, not exit clean — or vanish

**Symptom:** Triton ships no macOS wheels, so on the Mac dev box the suite initially either
error-collected (torch missing) or, after adding `pytest.importorskip`, skipped *everything*
— and pytest exits 5 for "no tests ran," which the repo's check hook rightly treats as
failure.

**Fix:** split the harness's torch-free logic into `bench/meta.py` with its own tests that
run on every machine. The suite always has a live core; kernel tests skip with a stated
reason only where torch/triton cannot exist. **Lesson:** an all-skip suite is
indistinguishable from a broken one — give every environment something real to assert.

## 2. PyPI's torch resolves to a wheel the Jetson driver can't load

**Symptom:** `pip install torch` (Jetson index as `--index-url`, PyPI as extra) resolved
torch 2.13.0+**cu130** — the SBSA aarch64 build. `jetson_smoke.py` step 2:
`torch.cuda.is_available() → False`, "driver too old (found version 12060)."

**Fix:** pin **by index, not just version**: `--index-url .../jp6/cu126 --no-deps
torch==2.11.0`, deps from PyPI separately. **Lesson:** on aarch64 there are now multiple
CUDA lineages of torch with the same package name; the resolver will happily pick a newer,
wrong one. The smoke test exists precisely to catch this before any benchmark exists.

## 3. The Jetson wheel links a library JetPack doesn't ship

**Symptom:** `import torch` → `libcudss.so.0: cannot open shared object file`.

**Fix:** `pip install nvidia-cudss-cu12` (PyPI carries an aarch64 wheel), plus
`scripts/jetson_env.sh` to put pip-installed NVIDIA lib dirs on `LD_LIBRARY_PATH` — they're
outside the wheel's RPATH. **Lesson:** "the wheel installed" and "the wheel's dynamic
dependencies resolve" are separate claims; `ldd` is part of bring-up.

## 4. Stale SBSA libraries shadowed the Jetson's cuBLAS — and hid until the first matmul

**Symptom** (`bench/results/session-20260818-065214.log`, committed): the full GPU test
suite ran — 40 tests passed, including every elementwise and softmax kernel — and all 9
matmul tests failed with `CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)`,
thrown by the *reference* `torch.matmul`, not the Triton kernel.

**Root cause:** fix #3's helper exports *every* `site-packages/nvidia/*/lib` dir, and the
venv still contained cu13/cu12 SBSA packages left behind by failure #2's torch (uninstalling
torch doesn't uninstall its dependency wheels). The SBSA cuBLAS shadowed the Jetson build
and died at first initialization — which happens at the first cuBLAS call in the process.
Everything before matmul was a false all-clear because nothing before matmul touches cuBLAS.

**Fix:** purge the venv to exactly one companion package (`nvidia-cudss-cu12`); warning
comment now lives in `jetson_env.sh`. **Lesson:** a broad `LD_LIBRARY_PATH` is a loaded gun,
and lazy library initialization means the blast radius reveals itself per-subsystem, not at
import. When only the cuBLAS-touching tests fail, suspect the linker before the kernel.

## 5. My correctness gate rejected a correct kernel

**Symptom** (`bench/results/session-20260818-070154.log`, committed): the fp32 matmul sweep
died at its first size — 0.5% of elements outside `rtol=atol=1e-2` vs an IEEE fp32
reference, max abs diff 4.2e-2 at 256³.

**Root cause:** not the kernel — the *gate*. `tl.dot` uses TF32 on Ampere; accumulation
error grows with K and differs by tiling order, so a fixed tolerance that passed at K≤128
(the unit tests) must fail at some larger K. A real indexing bug corrupts whole tiles, not
0.5% of elements.

**Fix:** the bench gate now measures **both** implementations against an fp64 reference and
requires my worst-case error within 3× cuBLAS's own (plus a small scale floor). cuBLAS runs
with TF32 enabled too, so the bar is like-for-like. **Lesson:** under reduced precision,
"close to a higher-precision reference" is the wrong question at scale; "no worse than the
vendor library under the same precision policy" is the defensible one.

## Ops footnote

The bench session wrapper (`scripts/bench_guss.sh`) stops resident services and self-healing
timers before benchmarking and restores them in a trap. Every failure above happened inside
that wrapper, and every time the box came back serving its LLM within seconds — the two
failed sessions' logs end with the same RESTORE lines as the successful ones. Benchmarks on
a machine with a day job are possible; unwound state is not optional.
