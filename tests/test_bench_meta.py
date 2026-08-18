"""Unit tests for the torch-free bench helpers. These run on every machine,
including ones where the kernel tests skip for lack of torch/triton."""

import pytest

from bench.meta import (
    device_slug,
    effective_gbps,
    parse_elementwise_sizes,
    parse_matmul_sizes,
    parse_softmax_sizes,
)


def test_parse_sizes_none_passthrough():
    assert parse_elementwise_sizes(None) is None
    assert parse_softmax_sizes(None) is None


def test_parse_elementwise_sizes():
    assert parse_elementwise_sizes("65536,262144") == [65536, 262144]
    assert parse_elementwise_sizes("1024") == [1024]


def test_parse_softmax_sizes():
    assert parse_softmax_sizes("4096x1024,1024x16384") == [(4096, 1024), (1024, 16384)]


def test_parse_matmul_sizes():
    assert parse_matmul_sizes(None) is None
    assert parse_matmul_sizes("512x512x512,1024x768x512") == [
        (512, 512, 512),
        (1024, 768, 512),
    ]


def test_parse_sizes_rejects_garbage():
    with pytest.raises(ValueError):
        parse_elementwise_sizes("4096x1024")
    with pytest.raises(ValueError):
        parse_softmax_sizes("4096")
    with pytest.raises(ValueError):
        parse_matmul_sizes("4096x1024")


def test_device_slug():
    assert device_slug("Orin") == "orin"
    assert device_slug("NVIDIA GeForce RTX 4090") == "nvidia-geforce-rtx-4090"
    assert device_slug("  Weird--Name!  ") == "weird--name"


def test_effective_gbps():
    # 1 GB moved in 1 second is 1 GB/s
    assert effective_gbps(1_000_000_000, 1000.0) == pytest.approx(1.0)
    # 4 bytes/element * 2^20 elements in 0.1 ms
    assert effective_gbps(4 * (1 << 20), 0.1) == pytest.approx(41.943, rel=1e-3)
