"""Create a comparable plot to Figure 11a (error vs. CNN width) next to the authors' data.

Paper curves are the final checkpoint of dd-cifar10-mcnn-p10-sgd-Mlist.
Our curves come from results-2.csv. Same overlay as plot_fig11b.py:
dashed and faded for the paper, solid with markers for our runs.
Both series are cut at width 80, the largest width in our grid. On the test
panel, each paper/ours pair also has unconnected markers for paper test error
minus our test error, at widths present in both.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# Viridis stops kept off the pale-yellow end so dashed lines stay readable.
# 12.5k is the green stop, 50k is the dark-purple stop.
_VIRIDIS = matplotlib.colormaps["viridis"]
COLORS = {n: _VIRIDIS(s) for n, s in zip((12500, 25000, 50000), (0.82, 0.50, 0.15))}

# Row order of each authors' Mlist. Train error rises with n, so this order is
# n=12500, 25000, 50000 for both the 10% and 20% noise releases.
PAPER_NS = [12500, 25000, 50000]
# Our grid stops at width 80 (1..40 step 1, then even widths through 80).
MAX_WIDTH = 128
# Widths from the authors' ks files, one per Mlist row.
PAPER_WIDTHS = {
    76: list(range(1, 65)) + [68, 72, 76, 80, 84, 88, 92, 96, 104, 112, 120, 128],
    60: (
        list(range(1, 33))
        + list(range(34, 65, 2))
        + [68, 72, 76, 80, 84, 88, 92, 96, 104, 112, 120, 128]
    ),
}


def load_paper(path):
    with path.open("rb") as f:
        blocks = pickle.load(f)
    if len(blocks) != len(PAPER_NS):
        raise SystemExit(f"{path} has {len(blocks)} curves, expected {len(PAPER_NS)}")
    rows = []
    for n, block in zip(PAPER_NS, blocks):
        test_error = block["Test Error"][:, -1]
        train_error = block["Train Error"][:, -1]
        widths = PAPER_WIDTHS.get(len(test_error))
        if widths is None:
            known = ", ".join(str(k) for k in sorted(PAPER_WIDTHS))
            raise SystemExit(
                f"n={n} has {len(test_error)} widths; expected one of {known}"
            )
        for size, te, tr in zip(widths, test_error, train_error):
            # Authors store error as a fraction; our csv stores percent.
            rows.append({
                "n": n,
                "size": int(size),
                "test_error": float(te) * 100,
                "train_error": float(tr) * 100,
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=SCRIPT_DIR / "results-2.csv")
    p.add_argument("--mlist", type=Path, default=SCRIPT_DIR / "dd-cifar10-mcnn-p10-sgd-Mlist")
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "fig11a")
    args = p.parse_args()

    ours = pd.read_csv(args.results) if args.results.exists() else None
    paper = load_paper(args.mlist) if args.mlist.exists() else None
    if ours is None and paper is None:
        raise SystemExit("nothing to plot")
    if ours is not None:
        ours = ours[ours["size"] <= MAX_WIDTH]
    if paper is not None:
        paper = paper[paper["size"] <= MAX_WIDTH]

    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(12, 4.5))
    for source, df, style in (("paper", paper, "--"), ("ours", ours, "-o")):
        if df is None:
            continue
        for n, g in df.groupby("n"):
            g = g.groupby("size", as_index=False).mean(numeric_only=True)
            kw = dict(color=COLORS.get(int(n)), markersize=3,
                      alpha=0.5 if source == "paper" else 1.0,
                      label=f"n={int(n)} ({source})")
            ax_test.plot(g["size"], g["test_error"], style, **kw)
            # 50k was the green train-error curve; leave it off this panel.
            if int(n) != 50000:
                ax_train.plot(g["size"], g["train_error"], style, **kw)

    # if ours is not None and paper is not None:
    #     merged = paper.merge(ours, on=["n", "size"], suffixes=("_paper", "_ours"))
    #     for n, g in merged.groupby("n"):
    #         g = g.sort_values("size")
    #         ax_test.plot(
    #             g["size"],
    #             g["test_error_paper"] - g["test_error_ours"],
    #             linestyle="none",
    #             marker="x",
    #             color=COLORS.get(int(n)),
    #             markersize=4,
    #             label=f"n={int(n)} (paper − ours)",
    #         )

    ax_test.set(xlabel="CNN width k", ylabel="Test error (%)",
                #title="Figure 11a: test error vs. width"
                )
    ax_train.set(xlabel="CNN width k", ylabel="Train error on noisy labels (%)",
                 #title="Interpolation threshold"
                 )
    for ax in (ax_test, ax_train):
        ax.axvspan(22, 40, facecolor="#b7e4c7", edgecolor="none", zorder=0)
        ax.set_xlim(0, MAX_WIDTH)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
