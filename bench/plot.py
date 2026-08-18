"""Render perf curves from committed benchmark CSVs.

    python -m bench.plot --results bench/results/<device-slug>

One PNG per CSV: GB/s vs size, one line per impl, IQR as a shaded band.
Needs matplotlib (not a core dependency): pip install matplotlib
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    by_impl: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_impl[row["impl"]].append(row)
    return by_impl


def main() -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib not installed: pip install matplotlib") from None

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="results/<device-slug> directory")
    args = ap.parse_args()

    results = Path(args.results)
    csvs = sorted(results.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {results}")

    for path in csvs:
        by_impl = load(path)
        sample = next(iter(by_impl.values()))[0]
        # matmul rows carry TFLOP/s (the meaningful metric there); others GB/s
        metric = "tflops" if sample.get("tflops") else "gbps"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for impl, rows in sorted(by_impl.items()):
            rows.sort(key=lambda r: int(r["numel"]))
            x = [int(r["numel"]) for r in rows]
            y = [float(r[metric]) for r in rows]
            ax.plot(x, y, marker="o", label=impl)
            if rows[0].get("ms_q20"):
                # IQR band recomputed from the timing quantiles (q80 = slower = lower rate)
                lo = [float(r[metric]) * float(r["ms_median"]) / float(r["ms_q80"]) for r in rows]
                hi = [float(r[metric]) * float(r["ms_median"]) / float(r["ms_q20"]) for r in rows]
                ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("elements")
        ax.set_ylabel("TFLOP/s" if metric == "tflops" else "effective GB/s")
        device = sample.get("device", "") or "device n/a"
        power = sample.get("power_mode", "") or "n/a"
        ax.set_title(f"{path.stem} — {device} ({power})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = path.with_suffix(".png")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
