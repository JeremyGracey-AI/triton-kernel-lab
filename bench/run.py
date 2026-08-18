"""Benchmark CLI. Requires a real CUDA device — the interpreter has no perf signal.

Usage (from the repo root):

    python -m bench.run --kernel elementwise --dtype fp16
    python -m bench.run --kernel softmax --dtype fp32 --sizes 4096x1024,1024x16384
    python -m bench.run --kernel all --out bench/results

Each run appends one CSV per (kernel, dtype) under results/<device-slug>/ and
writes env.json beside them. Every row carries the environment columns
(torch/triton versions, device, power mode) so a number can never be divorced
from the machine state that produced it.

Timing is triton.testing.do_bench: warmed up, median with an interquartile
range. A cooldown pause between sweep points keeps thermals from smearing
later points (this matters on a passively-constrained Jetson).
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

from bench.meta import (
    ELEMENTWISE_SIZES,
    MATMUL_SIZES,
    SOFTMAX_SIZES,
    device_slug,
    effective_gbps,
    parse_elementwise_sizes,
    parse_matmul_sizes,
    parse_softmax_sizes,
)
from kernels.elementwise import eager_mul_add_relu, fused_mul_add_relu
from kernels.matmul import matmul
from kernels.softmax import softmax, softmax_online

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def get_env_meta() -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda or "",
        "device": torch.cuda.get_device_name(0),
        "capability": "sm_{}{}".format(*torch.cuda.get_device_capability(0)),
        "platform": platform.platform(),
        "power_mode": "",
    }
    try:  # Jetson only; harmless elsewhere
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if "NV Power Mode" in line:
                meta["power_mode"] = line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return meta


def bench_ms(fn) -> tuple[float, float, float]:
    res = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    assert isinstance(res, list) and len(res) == 3, f"unexpected do_bench result: {res!r}"
    return res[0], res[1], res[2]


def rows_elementwise(dtype: torch.dtype, sizes: list[int], cooldown: float):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    for n in sizes:
        a, b, c = (torch.randn(n, device="cuda", dtype=dtype) for _ in range(3))
        torch.testing.assert_close(  # correctness gate precedes every timing
            fused_mul_add_relu(a, b, c).float(),
            torch.relu(a.float() * b.float() + c.float()),
            rtol=1e-3 if dtype == torch.float16 else 1.3e-6,
            atol=1e-3 if dtype == torch.float16 else 1e-5,
        )
        moved_bytes = 4 * n * itemsize  # 3 loads + 1 store
        for impl, fn in (
            ("triton_fused", lambda a=a, b=b, c=c: fused_mul_add_relu(a, b, c)),
            ("eager", lambda a=a, b=b, c=c: eager_mul_add_relu(a, b, c)),
        ):
            med, q20, q80 = bench_ms(fn)
            yield {
                "impl": impl,
                "shape": str(n),
                "numel": n,
                "ms_median": round(med, 5),
                "ms_q20": round(q20, 5),
                "ms_q80": round(q80, 5),
                "gbps": round(effective_gbps(moved_bytes, med), 2),
            }
        del a, b, c
        torch.cuda.empty_cache()
        time.sleep(cooldown)


def rows_softmax(dtype: torch.dtype, sizes: list[tuple[int, int]], cooldown: float):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    for n_rows, n_cols in sizes:
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
        ref = torch.softmax(x.float(), dim=-1)
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (None, None)
        torch.testing.assert_close(softmax(x).float(), ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(softmax_online(x).float(), ref, rtol=rtol, atol=atol)
        # Steady-state torch.compile baseline: fresh compile per shape (mirrors
        # Inductor's own shape specialization), compiled once here so do_bench
        # times the compiled artifact, never the compiler.
        compiled = torch.compile(lambda t: torch.softmax(t, dim=-1))
        torch.testing.assert_close(compiled(x).float(), ref, rtol=rtol, atol=atol)
        numel = n_rows * n_cols
        for impl, fn, passes in (
            ("triton_fused", lambda x=x: softmax(x), 2),  # read + write
            ("triton_online", lambda x=x: softmax_online(x), 3),  # 2 reads + write
            ("eager", lambda x=x: torch.softmax(x, dim=-1), 2),
            ("torch_compile", lambda x=x, c=compiled: c(x), 2),
        ):
            med, q20, q80 = bench_ms(fn)
            yield {
                "impl": impl,
                "shape": f"{n_rows}x{n_cols}",
                "numel": numel,
                "ms_median": round(med, 5),
                "ms_q20": round(q20, 5),
                "ms_q80": round(q80, 5),
                "gbps": round(effective_gbps(passes * numel * itemsize, med), 2),
            }
        del x, ref
        torch.cuda.empty_cache()
        time.sleep(cooldown)


def _matmul_error_gate(a: torch.Tensor, b: torch.Tensor) -> None:
    """Reduced-precision matmul (TF32, fp16) accumulates error that grows with
    K and differs by tiling order, so a fixed tolerance against an IEEE fp32
    reference is the wrong gate at large K — it rejected a correct kernel at
    256^3. Instead: measure BOTH implementations against an fp64 reference and
    require my worst-case error within a small factor of cuBLAS's own."""
    ref64 = torch.matmul(a.double(), b.double())
    scale = ref64.abs().max().item()
    err_mine = (matmul(a, b).double() - ref64).abs().max().item()
    err_cublas = (torch.matmul(a, b).double() - ref64).abs().max().item()
    limit = 3.0 * err_cublas + 1e-4 * scale
    if err_mine > limit:
        raise AssertionError(
            f"matmul gate: max abs err {err_mine:.3e} vs cuBLAS {err_cublas:.3e} "
            f"(limit {limit:.3e}, ref scale {scale:.3e})"
        )


def rows_matmul(dtype: torch.dtype, sizes: list[tuple[int, int, int]], cooldown: float):
    # My kernel's tl.dot uses TF32 for fp32 on Ampere; give cuBLAS the same
    # freedom so "% of cuBLAS" compares like for like (stated in the README).
    torch.backends.cuda.matmul.allow_tf32 = True
    itemsize = torch.tensor([], dtype=dtype).element_size()
    for m, n, k in sizes:
        a = torch.randn(m, k, device="cuda", dtype=dtype)
        b = torch.randn(k, n, device="cuda", dtype=dtype)
        # correctness gate — also triggers autotuning before any timing
        _matmul_error_gate(a, b)
        flops = 2.0 * m * n * k
        moved_bytes = (m * k + k * n + m * n) * itemsize
        results = {}
        for impl, fn in (
            ("cublas", lambda a=a, b=b: torch.matmul(a, b)),
            ("triton", lambda a=a, b=b: matmul(a, b)),
        ):
            med, q20, q80 = bench_ms(fn)
            results[impl] = med
            yield {
                "impl": impl,
                "shape": f"{m}x{n}x{k}",
                "numel": m * n,
                "ms_median": round(med, 5),
                "ms_q20": round(q20, 5),
                "ms_q80": round(q80, 5),
                "gbps": round(effective_gbps(moved_bytes, med), 2),
                "tflops": round(flops / (med * 1e-3) / 1e12, 3),
                "pct_cublas": round(100.0 * results["cublas"] / med, 1)
                if impl == "triton"
                else 100.0,
            }
        del a, b
        torch.cuda.empty_cache()
        time.sleep(cooldown)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kernel", choices=["elementwise", "softmax", "matmul", "all"], default="all"
    )
    ap.add_argument("--dtype", choices=[*DTYPES, "all"], default="all")
    ap.add_argument("--sizes", help="elementwise: N,N,...  softmax: RxC,RxC,...")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds between sweep points")
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args()

    if os.environ.get("TRITON_INTERPRET") == "1":
        raise SystemExit("refusing to benchmark the interpreter: unset TRITON_INTERPRET")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; benchmarks need real hardware")
    if args.sizes and args.kernel == "all":
        raise SystemExit("--sizes needs a single --kernel")

    meta = get_env_meta()
    out_dir = Path(args.out) / device_slug(meta["device"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(meta, indent=2) + "\n")

    kernels = ["elementwise", "softmax", "matmul"] if args.kernel == "all" else [args.kernel]
    dtypes = list(DTYPES) if args.dtype == "all" else [args.dtype]

    for kernel in kernels:
        for dtype_name in dtypes:
            dtype = DTYPES[dtype_name]
            if kernel == "elementwise":
                ew_sizes = parse_elementwise_sizes(args.sizes) or ELEMENTWISE_SIZES
                rows = rows_elementwise(dtype, ew_sizes, args.cooldown)
            elif kernel == "softmax":
                sm_sizes = parse_softmax_sizes(args.sizes) or SOFTMAX_SIZES
                rows = rows_softmax(dtype, sm_sizes, args.cooldown)
            else:
                mm_sizes = parse_matmul_sizes(args.sizes) or MATMUL_SIZES
                rows = rows_matmul(dtype, mm_sizes, args.cooldown)

            path = out_dir / f"{kernel}_{dtype_name}.csv"
            fields = [
                "kernel", "dtype", "impl", "shape", "numel",
                "ms_median", "ms_q20", "ms_q80", "gbps", "tflops", "pct_cublas",
                "torch", "triton", "cuda", "device", "capability", "power_mode",
                "timestamp_utc",
            ]
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, restval="")
                writer.writeheader()
                for row in rows:
                    row.update(
                        kernel=kernel,
                        dtype=dtype_name,
                        **{k: meta[k] for k in (
                            "torch", "triton", "cuda", "device", "capability",
                            "power_mode", "timestamp_utc",
                        )},
                    )
                    writer.writerow(row)
                    extra = f"  {row['tflops']:>7.2f} TFLOP/s" if "tflops" in row else ""
                    print(
                        f"{kernel:<11} {dtype_name} {row['impl']:<13} {row['shape']:>14}"
                        f"  {row['ms_median']:>9.4f} ms  {row['gbps']:>7.1f} GB/s{extra}"
                    )
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
