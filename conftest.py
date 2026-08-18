"""Test configuration.

Tests default to Triton's CPU interpreter (TRITON_INTERPRET=1) so the suite
runs anywhere — CI included — with no GPU. This is a correctness gate only;
the interpreter executes programs sequentially via numpy and carries no
performance signal. On a CUDA box, run the same suite against the real
compiler with:  TRITON_INTERPRET=0 pytest

The env var must be set before triton is first imported, which is why it
lives here: pytest loads this conftest before any test module.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(scope="session")
def device() -> str:
    return "cpu" if _interpreting() else "cuda"
