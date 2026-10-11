"""Fig. 21 check: our lambda = 0 CNN runs on 10%-noisy test labels vs. the authors' released run."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import GOLD, HERE, INK, MUTED, PURPLE, authors, save

RELEASED = "pct-cifar10-mcnn-mdd-50k-p10-sgd-aug-fast"
EPOCHS = (1280, 1459)
EP_COLORS = {1280: GOLD, 1459: PURPLE}
K_EPOCH_PANEL = 32


def load(results, partial):
    runs = {}
    for d in sorted((results / "E1n").glob("cnn5_k*")):
        met = d / "metrics.csv"
        if not met.exists() or (not partial and not (d / "summary.json").exists()):
            continue
        m = pd.read_csv(met)
        if "test_err_noisy" in m and len(m):
            runs[int(d.name.split("_k")[1].split("_")[0])] = m.set_index("epoch")
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()

    runs = load(HERE / "results", args.partial)
    a = authors(RELEASED)
    if not runs or a is None:
        raise SystemExit("no E1n runs or no released data (run authors_data.py)")
    ks_a = [int(k) for k in a["ks"]]

    rows = []
    for k, m in sorted(runs.items()):
        if k not in ks_a:
            continue
        i = ks_a.index(k)
        for ep in EPOCHS:
            if ep not in m.index:
                continue
            r = m.loc[ep]
            rows.append({"k": k, "epoch": ep, "err_clean": r.test_err, "err_noisy": r.test_err_noisy,
                         "loss_clean": r.test_loss, "loss_noisy": r.test_loss_noisy,
                         "released_err": float(a["test_error"][0, i, ep - 1]),
                         "released_loss": float(a["test_loss"][0, i, ep - 1])})
    df = pd.DataFrame(rows)
    (HERE / "tables").mkdir(exist_ok=True)
    df.to_csv(HERE / "tables" / "e1n_check.csv", index=False, float_format="%.4f")

    e1 = {}
    for d in (HERE / "results" / "E1").glob("cnn5_k*_wd0e+00_sgd_s0_n0"):
        if (d / "summary.json").exists():
            e1[int(d.name.split("_k")[1].split("_")[0])] = pd.read_csv(d / "metrics.csv").test_err.iloc[-1]
    for ep in EPOCHS:
        s = df[df.epoch == ep]
        if s.empty:
            continue
        de, dl = 100 * (s.err_noisy - s.released_err), s.loss_noisy - s.released_loss
        big = s.k >= 28
        print(f"epoch {ep} ({len(s)} widths): noisy error - released {de.mean():+.1f} +- {de.std():.1f} pp | "
              f"noisy loss - released {dl.mean():+.2f} (k>=28: {dl[big].mean():+.2f}, k<28: {dl[~big].mean():+.2f})")
        if ep == 1280 and e1:
            same = s[s.k.isin(e1)]
            diff = 100 * (same.err_clean - same.k.map(e1))
            print(f"  rerun vs original E1 (clean error, epoch 1280): {diff.mean():+.1f} +- {diff.std():.1f} pp")
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    for ep in EPOCHS:
        s = df[df.epoch == ep].sort_values("k")
        if s.empty:
            continue
        c = EP_COLORS[ep]
        for ax, ours, theirs in ((axes[0], "err_noisy", "released_err"), (axes[1], "loss_noisy", "released_loss")):
            ax.plot(s.k, s[ours], "-o", color=c, label=f"ours, epoch {ep}")
            ax.plot(s.k, s[theirs], "--", color=c, lw=1.1, label=f"Nakkiran et al., epoch {ep}")
    last = df[df.epoch == df.epoch.max()].sort_values("k")
    for ax, col in ((axes[0], "err_clean"), (axes[1], "loss_clean")):
        ax.plot(last.k, last[col], ":", color=MUTED, label=f"ours, clean test labels, epoch {last.epoch.max()}")
        ax.set_xlabel("CNN width parameter k")
    axes[0].set_ylabel("Test error")
    axes[1].set_ylabel("Test loss")
    axes[0].set_title("(a) error, noisy test labels")
    axes[1].set_title("(b) loss, noisy test labels")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    ax = axes[2]
    if K_EPOCH_PANEL in runs and K_EPOCH_PANEL in ks_a:
        m = runs[K_EPOCH_PANEL]
        rel = a["test_loss"][0, ks_a.index(K_EPOCH_PANEL)]
        ax.plot(np.arange(1, len(rel) + 1), rel, "--", color=INK, lw=1.0, label="Nakkiran et al.")
        ax.plot(m.index, m.test_loss_noisy, "-", color=PURPLE, lw=1.0, label="ours")
        for ep in EPOCHS:
            ax.axvline(ep, color=EP_COLORS[ep], lw=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.legend(loc="upper left", fontsize=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss (noisy labels)")
    ax.set_title(f"(c) k = {K_EPOCH_PANEL} over training")
    save(fig, "fig21_noisy_check", HERE / "figures")


if __name__ == "__main__":
    main()
