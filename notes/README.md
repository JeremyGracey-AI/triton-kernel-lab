# Study track

One module at a time, each with a concrete artifact produced from *this repo's own kernels* —
no artifact, no module. Modules land as they're finished (see the roadmap in the root README).

| Module | Topic | Artifact |
|---|---|---|
| A | GPU execution model: SIMT, memory hierarchy, occupancy | occupancy + bandwidth worksheet for the elementwise kernel on sm_87 |
| B | Triton's model: blocked programs vs CUDA threads (MAPL '19 paper) | annotated comparison against a raw CUDA version of the same kernel |
| C | LLVM foundations: SSA, IR structure, pass pipelines | annotated LLVM-IR + PTX from the elementwise kernel |
| D | Triton's lowering pipeline | softmax traced and annotated through TTIR → TTGIR → LLVM IR → PTX |
| E | TorchDynamo/Inductor: bytecode capture, guards, graph breaks, FX, AOTAutograd | what `torch.compile` does to this repo's softmax + steady-state compile baseline |
