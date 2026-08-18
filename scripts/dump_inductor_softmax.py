"""Capture the Triton kernel Inductor generates for softmax, to read against mine.

Run on a CUDA box (Inductor only emits Triton when targeting CUDA):

    TORCH_LOGS=output_code python scripts/dump_inductor_softmax.py 2> inductor_softmax_raw.txt

The generated module lands on stderr via the output_code artifact log; the
annotated read lives in notes/e-inductor/.
"""

import torch


def softmax_fp16(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def main() -> None:
    assert torch.cuda.is_available(), "needs CUDA — Inductor emits C++/OpenMP on CPU"
    compiled = torch.compile(softmax_fp16)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
    y = compiled(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        y.float(), torch.softmax(x.float(), dim=-1), rtol=1e-3, atol=1e-3
    )
    print("compiled softmax ran and matched fp32 reference")


if __name__ == "__main__":
    main()
