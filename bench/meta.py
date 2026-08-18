"""Pure helpers for the bench harness: no torch/triton imports.

Split out so the logic is unit-testable on machines that can't install
torch/triton (macOS), where the kernel tests skip.
"""

ELEMENTWISE_SIZES = [1 << p for p in (16, 18, 20, 22, 24, 26)]
SOFTMAX_SIZES = [(4096, 256), (4096, 1024), (4096, 4096), (16384, 1024), (1024, 16384)]
MATMUL_SIZES = [(s, s, s) for s in (256, 512, 1024, 2048, 4096)]


def parse_elementwise_sizes(raw: str | None) -> list[int] | None:
    """"65536,262144" → [65536, 262144]"""
    if raw is None:
        return None
    return [int(s) for s in raw.split(",")]


def parse_softmax_sizes(raw: str | None) -> list[tuple[int, int]] | None:
    """"4096x1024,1024x256" → [(4096, 1024), (1024, 256)]"""
    if raw is None:
        return None
    sizes = []
    for chunk in raw.split(","):
        rows, cols = chunk.split("x")
        sizes.append((int(rows), int(cols)))
    return sizes


def parse_matmul_sizes(raw: str | None) -> list[tuple[int, int, int]] | None:
    """"512x512x512,1024x768x512" → [(512, 512, 512), (1024, 768, 512)]"""
    if raw is None:
        return None
    sizes = []
    for chunk in raw.split(","):
        m, n, k = chunk.split("x")
        sizes.append((int(m), int(n), int(k)))
    return sizes


def device_slug(device_name: str) -> str:
    """"Orin" → "orin"; "NVIDIA GeForce RTX 4090" → "nvidia-geforce-rtx-4090"."""
    return "".join(ch if ch.isalnum() else "-" for ch in device_name.lower()).strip("-")


def effective_gbps(bytes_moved: int, ms: float) -> float:
    """Effective bandwidth for a kernel that moved bytes_moved in ms milliseconds."""
    return bytes_moved / (ms * 1e-3) / 1e9
