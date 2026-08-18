# Study track

One module at a time, each with a concrete artifact produced from *this repo's own kernels* —
no artifact, no module. Modules land as they're finished (see the roadmap in the root README).

| Module | Topic | Artifact |
|---|---|---|
| A | GPU execution model: SIMT, memory hierarchy, occupancy | **[landed](a-gpu-execution/README.md)**: occupancy + bandwidth worksheet for the elementwise kernel on sm_87, real register counts via cuobjdump |
| B | Triton's model: blocked programs vs CUDA threads (MAPL '19 paper) | **[landed](b-triton-model/README.md)**: the repo's own Triton/CUDA matched pair, measured, plus the one-source-two-binaries specialization receipt |
| C | LLVM foundations: SSA, IR structure, pass pipelines | **[landed](c-llvm/README.md)**: annotated LLVM-IR + PTX from the elementwise kernel; vectorized vs ragged specialization committed side by side |
| D | Triton's lowering pipeline | **[landed](d-lowering/README.md)**: softmax traced and annotated through TTIR → TTGIR → LLVM IR → PTX, from a real Orin compile |
| E | TorchDynamo/Inductor: bytecode capture, guards, graph breaks, FX, AOTAutograd | **[seeded](e-inductor/README.md)**: Inductor's generated softmax read against mine, from a real Orin compile; steady-state compile baseline lands with M4 |
