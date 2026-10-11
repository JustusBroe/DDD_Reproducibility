"""Quantitative comparison of our runs with the paper's raw curves.

For each run (and the matching paper curve) it reports, on a 5-epoch moving
average of the test error:
  * peak            point where the test error has risen most above its running minimum
  * first minimum   that running minimum (lowest test error before the peak), and its epoch
  * final           mean test error over the last 1 % of training
  * rise            peak - first minimum   (> 0: test error goes up again)
  * 2nd descent     peak - final           (> 0: test error comes back down)
  * interp. epoch   first epoch with train error < 1 % (noisy labels)
and, for our runs only, the epoch at which half of the mislabeled samples are memorised.

    python summarize.py            # -> results/summary.csv, results/summary.tex
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from plot_epochwise import HERE, REFS, RUN_RE


def smooth(y, w=5):
    return pd.Series(y).rolling(w, center=True, min_periods=1).mean().to_numpy()


def describe(epochs, test_err, train_epochs, train_err):
    """Double-descent statistics of one test-error curve (epochs are 1-based)."""
    te = smooth(test_err)
    below = np.flatnonzero(np.asarray(train_err) < 0.01)
    interp = float(train_epochs[below[0]]) if len(below) else np.nan
    # Peak = largest rise of the test error above its best value so far; the first minimum is
    # that best value. A monotone curve gives a rise of ~0 (noise only).
    i_peak = int(np.argmax(te - np.minimum.accumulate(te)))
    i_min = int(np.argmin(te[:i_peak + 1]))
    final = float(np.mean(test_err[-max(1, len(test_err) // 100):]))
    return {"first_min": te[i_min], "first_min_ep": int(epochs[i_min]),
            "peak": te[i_peak], "peak_ep": int(epochs[i_peak]),
            "final": final, "rise": te[i_peak] - te[i_min], "second_descent": te[i_peak] - final,
            "interp_ep": interp, "epochs": int(epochs[-1])}


def ours(log, cut=None):
    df = pd.read_csv(log)
    if cut is not None:
        df = df[df["epoch"] <= cut]
    tr = df.dropna(subset=["full_train_err"])
    stats = describe(df["epoch"].to_numpy(), df["test_err"].to_numpy(),
                     tr["epoch"].to_numpy(), tr["full_train_err"].to_numpy())
    half = tr[tr["noisy_fit"] >= 0.5]
    stats["memorised_half_ep"] = int(half["epoch"].iloc[0]) if len(half) else np.nan
    return stats


def paper(arch, dataset, noise, k, cut=None):
    folder, entry = REFS.get((arch, dataset), {}).get(noise, (None, None))
    path = HERE / "reference" / str(folder) / "curves.npz"
    if folder is None or not path.exists():
        return None
    z = np.load(path)
    hit = np.flatnonzero(z["ks"] == k)
    if not len(hit):
        return None
    te, tr = z[f"{entry}__Test Error"][hit[0]], z[f"{entry}__Train Error"][hit[0]]
    epochs = np.arange(1, len(te) + 1)
    if cut is not None:
        keep = epochs <= cut
        epochs, te, tr = epochs[keep], te[keep], tr[keep]
    return describe(epochs, te, epochs, tr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=HERE / "runs")
    p.add_argument("--out", type=Path, default=HERE / "results")
    a = p.parse_args()

    rows = []
    for log in sorted(a.runs.glob("*/log.csv")):
        m = RUN_RE.match(log.parent.name)
        if not m:
            continue
        key = dict(arch=m["arch"], dataset=m["dataset"], k=int(m["k"]), noise=int(m["p"]), seed=int(m["seed"]))
        o = ours(log)
        # Compare over the same training length we actually ran.
        ref = paper(m["arch"], m["dataset"], int(m["p"]), int(m["k"]), cut=o["epochs"])
        rows.append(key | {"source": "ours"} | o)
        if ref is not None:
            rows.append(key | {"source": "paper"} | ref)
    if not rows:
        raise SystemExit("no runs found")

    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "summary.csv", index=False, float_format="%.4f")

    cols = ["arch", "dataset", "k", "noise", "source", "first_min", "first_min_ep", "peak", "peak_ep",
            "final", "rise", "second_descent", "interp_ep"]
    tex = df[cols].rename(columns={
        "arch": "Model", "dataset": "Data", "noise": "$p$ (\\%)", "source": "Source",
        "first_min": "Min.", "first_min_ep": "@ep", "peak": "Peak", "peak_ep": "@ep ", "final": "Final",
        "rise": "$\\Delta_{\\uparrow}$", "second_descent": "$\\Delta_{\\downarrow}$", "interp_ep": "Interp."})
    (a.out / "summary.tex").write_text(tex.to_latex(index=False, float_format="%.3f", na_rep="--"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols + ["memorised_half_ep"]].round(3).to_string(index=False))
    print(f"\nwrote {a.out / 'summary.csv'} and {a.out / 'summary.tex'}")


if __name__ == "__main__":
    main()
