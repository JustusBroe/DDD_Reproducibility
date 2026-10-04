"""Create comparable plot to Figure 11b (test loss vs. number of train samples) next to the authors' data.
For curiousity also include the test_error (not only loss) to see whether their EMC-idea checks out here.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))
COLORS = {64: "tab:blue", 80: "tab:orange"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "results.csv")
    p.add_argument("--paper-csv", type=Path, nargs="*",
                   default=sorted(PROJECT_ROOT.glob("dd-translation-samples-*.csv")))
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "fig11b")
    args = p.parse_args()

    ours = pd.read_csv(args.results) if args.results.exists() else None
    paper = pd.concat([pd.read_csv(f) for f in args.paper_csv]) if args.paper_csv else None
    if ours is None and paper is None:
        raise SystemExit("nothing to plot")

    fig, (ax_loss, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))
    for source, df, style in (("paper", paper, "--"), ("ours", ours, "-o")):
        if df is None:
            continue
        for size, g in df.groupby("size"):
            g = g.groupby("n", as_index=False).mean(numeric_only=True)
            kw = dict(color=COLORS.get(int(size)), markersize=3,
                      alpha=0.5 if source == "paper" else 1.0,
                      label=f"d_model={int(size)} ({source})")
            ax_loss.plot(g["n"] / 1000, g["test_loss"], style, **kw)
            ax_err.plot(g["n"] / 1000, g["train_error"], style, **kw)

    ax_loss.set(xlabel="Number of train samples (thousands)", ylabel="Test loss (per-token NLL, bits)",
                title="Figure 11b: sample-wise non-monotonicity")
    ax_err.set(xlabel="Number of train samples (thousands)", ylabel="Train error (%)",
               title="Interpolation threshold")
    ax_err.axhline(1.0, color="grey", lw=0.8, ls=":")
    for ax in (ax_loss, ax_err):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
