"""Correctness gate for the tiled matmul.

Reference is fp32 torch.matmul of fp32-cast inputs. Tolerances are loose
enough to admit TF32 accumulation on Ampere-class GPUs (~1e-3 relative) and
fp16 output rounding; the interpreter computes in fp32 numpy and sits well
inside them. Sizes deliberately include ragged shapes that exercise the
store mask and the K-tail load mask.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed here; suite runs in CI/Linux")
pytest.importorskip("triton", reason="triton not installed here; suite runs in CI/Linux")

from kernels.matmul import matmul

TOLS = {
    torch.float32: (1e-2, 1e-2),
    torch.float16: (1e-2, 1e-2),
}


def _ref_fp32(a, b):
    return torch.matmul(a.float(), b.float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 16, 16),  # single tile, exact fit
        (64, 64, 48),  # K-tail (48 = 3 x BLOCK_K in interpreter config)
        (33, 65, 37),  # ragged everything: store mask + K mask + wrap trick
        (128, 96, 128),
    ],
)
def test_matches_fp32_reference(device, dtype, m, n, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    rtol, atol = TOLS[dtype]
    torch.testing.assert_close(matmul(a, b).float(), _ref_fp32(a, b), rtol=rtol, atol=atol)


def test_non_contiguous_inputs(device):
    torch.manual_seed(1)
    a = torch.randn(48, 32, device=device).t()  # 32x48 view, non-contiguous
    b = torch.randn(48, 24, device=device)
    out = matmul(a, b)
    assert out.shape == (32, 24)
    torch.testing.assert_close(out, _ref_fp32(a, b), rtol=1e-2, atol=1e-2)


def test_shape_and_dtype_errors(device):
    a = torch.randn(8, 4, device=device)
    with pytest.raises(ValueError, match="inner dims"):
        matmul(a, torch.randn(8, 4, device=device))
    with pytest.raises(ValueError, match="2-D"):
        matmul(a, torch.randn(4, 4, 4, device=device))
    with pytest.raises(ValueError, match="dtype"):
        matmul(a, torch.randn(4, 8, device=device, dtype=torch.float16))
