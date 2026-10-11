"""Test-error difference (paper minus our runs) as three point clouds.

One panel per training-set size. Each panel has two least-squares lines in CNN
width: one while our train error is still above 1%, and one after it has fallen
to that level (essentially interpolated). Points are the shared widths up to 80.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_fig11a import COLORS, MAX_WIDTH, PAPER_NS, SCRIPT_DIR, load_paper

# Our train error is "almost 0" / "very small" once it drops to this percent.
SMALL_TRAIN_ERROR = 1.0


def fit_label(prefix, intercept, slope):
    sign = "+" if slope >= 0 else "−"
    return f"{prefix}: {intercept:.2f} {sign} {abs(slope):.3f} c"


def fit_segment(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=SCRIPT_DIR / "results-2.csv")
    p.add_argument("--mlist", type=Path, default=SCRIPT_DIR / "dd-cifar10-mcnn-p10-sgd-Mlist")
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "fig11a_diff")
    args = p.parse_args()

    if not args.results.exists() or not args.mlist.exists():
        raise SystemExit("need both results-2.csv and the paper Mlist")

    ours = pd.read_csv(args.results)
    ours = ours[ours["size"] <= MAX_WIDTH]
    paper = load_paper(args.mlist)
    paper = paper[paper["size"] <= MAX_WIDTH]
    merged = paper.merge(ours, on=["n", "size"], suffixes=("_paper", "_ours"))
    merged["diff"] = merged["test_error_paper"] - merged["test_error_ours"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, n in zip(axes, PAPER_NS):
        g = merged[merged["n"] == n].sort_values("size")
        x = g["size"].to_numpy(dtype=float)
        y = g["diff"].to_numpy(dtype=float)
        train = g["train_error_ours"].to_numpy(dtype=float)
        color = COLORS[int(n)]
        small = np.flatnonzero(train <= SMALL_TRAIN_ERROR)
        if len(small) == 0 or small[0] == 0:
            raise SystemExit(f"n={int(n)} has no split where train error becomes small")
        split_at = int(small[0])
        split_width = x[split_at]
        ax.plot(x, y, linestyle="none", marker="x", color=color, markersize=4,
                label="paper − ours")
        ax.axvline(split_width, color="0.45", ls=":", lw=1, zorder=1)
        for mask, style, name in (
            (np.arange(len(x)) < split_at, "-", f"c < {split_width:.0f}"),
            (np.arange(len(x)) >= split_at, "--", f"c ≥ {split_width:.0f}"),
        ):
            slope, intercept = fit_segment(x[mask], y[mask])
            x_line = np.array([x[mask].min(), x[mask].max()])
            label = fit_label(name, intercept, slope)
            ax.plot(x_line, intercept + slope * x_line, style, color="black", lw=1.2,
                    label=label)
            print(f"n={int(n)} {label}  ({mask.sum()} points)")
        ax.set_xlim(0, MAX_WIDTH)
        ax.set_xlabel("CNN width c")
        ax.set_title(f"n={int(n)}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)

    axes[0].set_ylabel("Test error difference, paper − ours (pp)")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
