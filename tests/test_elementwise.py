"""Correctness gate for the fused mul-add-relu kernel.

Reference is computed in fp32 and compared in fp32 with dtype-scaled
tolerances, so fp16 rounding in the kernel is judged against fp16
expectations, not fp32 ones.
"""

import pytest

# On machines without torch/triton (e.g. macOS: Triton ships no darwin wheels)
# the suite skips with a visible reason instead of erroring; CI and any Linux
# environment install both and run everything for real.
torch = pytest.importorskip("torch", reason="torch not installed here; suite runs in CI/Linux")
pytest.importorskip("triton", reason="triton not installed here; suite runs in CI/Linux")

from kernels.elementwise import eager_mul_add_relu, fused_mul_add_relu

# (rtol, atol) per input dtype, applied to a comparison done in fp32.
TOLS = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-3, 1e-3),
}


def _ref_fp32(a, b, c):
    return torch.relu(a.float() * b.float() + c.float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n", [1, 17, 1024, 4099])
def test_matches_fp32_reference(device, dtype, n):
    torch.manual_seed(0)
    a, b, c = (torch.randn(n, device=device, dtype=dtype) for _ in range(3))
    out = fused_mul_add_relu(a, b, c)
    rtol, atol = TOLS[dtype]
    torch.testing.assert_close(out.float(), _ref_fp32(a, b, c), rtol=rtol, atol=atol)


def test_matches_eager_baseline(device):
    torch.manual_seed(1)
    a, b, c = (torch.randn(2048, device=device) for _ in range(3))
    torch.testing.assert_close(fused_mul_add_relu(a, b, c), eager_mul_add_relu(a, b, c))


def test_2d_input_and_noncontiguous(device):
    torch.manual_seed(2)
    a = torch.randn(64, 33, device=device).t()  # non-contiguous view
    b = torch.randn(33, 64, device=device)
    c = torch.randn(33, 64, device=device)
    out = fused_mul_add_relu(a, b, c)
    assert out.shape == (33, 64)
    torch.testing.assert_close(out.float(), _ref_fp32(a, b, c))


def test_shape_mismatch_raises(device):
    a = torch.randn(8, device=device)
    b = torch.randn(9, device=device)
    with pytest.raises(ValueError, match="shape mismatch"):
        fused_mul_add_relu(a, b, a)


def test_small_block_size_covers_tail(device):
    torch.manual_seed(3)
    a, b, c = (torch.randn(130, device=device) for _ in range(3))
    out = fused_mul_add_relu(a, b, c, block_size=64)  # 130 = 2 full blocks + tail of 2
    torch.testing.assert_close(out.float(), _ref_fp32(a, b, c))
