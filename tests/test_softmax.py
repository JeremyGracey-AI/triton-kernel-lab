"""Correctness gate for both softmax variants.

Comparison happens in fp32 against torch.softmax of the fp32-cast input
(the plan's rule: softmax is always compared in fp32). The online variant
is additionally forced through multi-tile paths with tile sizes smaller
than the row.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed here; suite runs in CI/Linux")
pytest.importorskip("triton", reason="triton not installed here; suite runs in CI/Linux")

from kernels.softmax import softmax, softmax_online

TOLS = {
    torch.float32: (1.3e-6, 1e-6),
    torch.float16: (1e-3, 1e-3),
}


def _ref_fp32(x):
    return torch.softmax(x.float(), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1), (4, 17), (32, 128), (8, 1000)])
def test_single_block_matches_reference(device, dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device, dtype=dtype)
    rtol, atol = TOLS[dtype]
    torch.testing.assert_close(softmax(x).float(), _ref_fp32(x), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("shape", "tile_n"),
    [
        ((4, 17), 8),  # tail tile narrower than TILE_N
        ((3, 1000), 64),  # many tiles per row
        ((2, 4099), 1024),  # wide row, non-power-of-2
        ((5, 33), 64),  # single tile wider than the row
    ],
)
def test_online_matches_reference(device, dtype, shape, tile_n):
    torch.manual_seed(1)
    x = torch.randn(*shape, device=device, dtype=dtype)
    rtol, atol = TOLS[dtype]
    torch.testing.assert_close(
        softmax_online(x, tile_n=tile_n).float(), _ref_fp32(x), rtol=rtol, atol=atol
    )


def test_variants_agree(device):
    torch.manual_seed(2)
    x = torch.randn(16, 257, device=device)
    torch.testing.assert_close(softmax(x), softmax_online(x, tile_n=64))


def test_rows_sum_to_one(device):
    torch.manual_seed(3)
    x = torch.randn(8, 300, device=device)
    sums = softmax(x).sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


def test_stable_under_large_inputs(device):
    # Without max-subtraction exp() overflows here; the row should still be finite.
    x = torch.full((2, 64), 3.0e4, device=device)
    out = softmax(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2, device=device))


def test_non_2d_raises(device):
    with pytest.raises(ValueError, match="2-D"):
        softmax(torch.randn(4, 4, 4, device=device))
