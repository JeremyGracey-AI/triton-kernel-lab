"""Module E artifact: what Dynamo actually captures for softmax — and what breaks it.

Runs torch._dynamo.explain on two functions: a clean softmax (expect one graph, zero
breaks) and a deliberately hostile variant with a .item() data-dependency mid-function
(expect a graph break splitting it in two). Prints graph counts, break reasons, and the
guards Dynamo installed. Run on a CUDA box; commit the output beside module E:

    python scripts/dynamo_explain_softmax.py > notes/e-inductor/dynamo_explain.txt
"""

import torch
import torch._dynamo as dynamo


def clean_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def breaking_softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.softmax(x, dim=-1)
    # .item() forces a GPU->CPU sync and a Python-value data dependency:
    # Dynamo cannot trace through it, so the function splits into two graphs.
    peak = y.max().item()
    return y * (1.0 if peak > 0 else -1.0)


def report(name, fn, x):
    print(f"{'=' * 70}\n{name}\n{'=' * 70}")
    ex = dynamo.explain(fn)(x)
    print(f"graphs: {ex.graph_count}  graph breaks: {ex.graph_break_count}  ops: {ex.op_count}")
    for i, reason in enumerate(ex.break_reasons):
        print(f"break[{i}]: {reason.reason}")
    for i, graph in enumerate(ex.graphs):
        print(f"\n--- captured FX graph {i} ---")
        graph.graph.print_tabular()
    print()


def main() -> None:
    assert torch.cuda.is_available()
    x = torch.randn(8, 64, device="cuda", dtype=torch.float16)
    report("clean softmax (expect 1 graph, 0 breaks)", clean_softmax, x)
    dynamo.reset()
    report("softmax with .item() mid-function (expect a graph break)", breaking_softmax, x)


if __name__ == "__main__":
    main()
