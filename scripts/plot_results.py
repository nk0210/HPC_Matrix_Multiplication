#!/usr/bin/env python3
"""Generate the Section 8 graphs from results/benchmark.csv.

Reads the raw benchmark CSV (three runs per configuration), collapses the
runs to their median per (matrix_size, threads) and writes two PNGs into
plots/:

  * speedup_vs_threads.png    - one line per matrix size, plus an ideal
                                linear-speedup reference line (y = x).
  * efficiency_vs_threads.png - one line per matrix size.

Usage:
    python scripts/plot_results.py
"""

import os
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "benchmark.csv")
PLOTS_DIR = os.path.join(ROOT, "plots")


def load_median(csv_path):
    """Return a frame with the median of the runs per (matrix_size, threads)."""
    df = pd.read_csv(csv_path)
    required = {
        "matrix_size",
        "threads",
        "time_seq",
        "time_par",
        "speedup",
        "efficiency",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {sorted(missing)}")

    median = (
        df.groupby(["matrix_size", "threads"], as_index=False)[
            ["time_seq", "time_par", "speedup", "efficiency"]
        ]
        .median()
        .sort_values(["matrix_size", "threads"])
    )
    return median


def plot_speedup(median, out_path):
    sizes = sorted(median["matrix_size"].unique())
    threads = sorted(median["threads"].unique())

    plt.figure(figsize=(8, 6))
    for size in sizes:
        sub = median[median["matrix_size"] == size]
        plt.plot(sub["threads"], sub["speedup"], marker="o", label=f"N = {size}")

    # Ideal linear speedup: speedup == number of threads.
    plt.plot(threads, threads, "k--", linewidth=1.2, label="Ideal (linear)")

    plt.title("Speedup vs Threads")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup  (time_seq / time_par)")
    plt.xticks(threads)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("wrote", out_path)


def plot_efficiency(median, out_path):
    sizes = sorted(median["matrix_size"].unique())
    threads = sorted(median["threads"].unique())

    plt.figure(figsize=(8, 6))
    for size in sizes:
        sub = median[median["matrix_size"] == size]
        plt.plot(
            sub["threads"], sub["efficiency"], marker="o", label=f"N = {size}"
        )

    plt.axhline(100.0, color="k", linestyle="--", linewidth=1.2, label="Ideal (100%)")

    plt.title("Parallel Efficiency vs Threads")
    plt.xlabel("Number of threads")
    plt.ylabel("Efficiency  (%)")
    plt.xticks(threads)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("wrote", out_path)


def main():
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(
            f"{CSV_PATH} not found. Run 'make benchmark' (or scripts/benchmark.sh) first."
        )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    median = load_median(CSV_PATH)

    print("Median results per (matrix_size, threads):")
    print(median.to_string(index=False))
    print()

    plot_speedup(median, os.path.join(PLOTS_DIR, "speedup_vs_threads.png"))
    plot_efficiency(median, os.path.join(PLOTS_DIR, "efficiency_vs_threads.png"))


if __name__ == "__main__":
    sys.exit(main())
