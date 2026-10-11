"""Plot epoch-wise double descent: our runs (solid) vs the paper's raw curves (dashed).

    python plot_epochwise.py --mode widths --noise 20    # Fig. 9 left (+ Fig. 9 right heatmap from paper data)
    python plot_epochwise.py --mode fig10                # Fig. 10 a-c: one model, noise 0/10/20 %, test + train
    python plot_epochwise.py --mode all

Paper curves come from reference/ (run fetch_reference.py first); panels are drawn
with whatever of ours / paper data exists, so they can be made while runs are going.
Figures go to figures/*.{png,pdf}.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF_DIR = HERE / "reference"
# One fixed colour per width / noise level, assigned in order, so it is stable across figures.
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
NOISES = [0, 10, 20]

# (arch, dataset) -> {noise %: (reference folder, Mlist entry)}. Entry order was inferred from final
# test error (higher noise -> higher error) and checked against the published Fig. 10a/b.
# No public file matches the published Fig. 10c (CNN): its epoch-wise data sits in the authors' private
# bucket, and the public CNN/SGD files disagree with that panel. The CNN is compared with the published
# figure in the report instead.
REFS = {
    ("resnet18k", "cifar10"): {p: ("cifar10-resnet18k-50k-adam", i) for i, p in enumerate(NOISES)},
    ("resnet18k", "cifar100"): {p: ("cifar100-resnet18k-50k-adam", i) for i, p in enumerate(NOISES)},
}
FIG10_PANELS = [("resnet18k", "cifar10", "adam", 64, "(a) ResNet18 on CIFAR-10"),
                ("resnet18k", "cifar100", "adam", 64, "(b) ResNet18 on CIFAR-100"),
                ("mcnn", "cifar10", "sgd", 128, "(c) 5-layer CNN on CIFAR-10")]
RUN_RE = re.compile(r"(?P<arch>resnet18k|mcnn)_(?P<dataset>cifar10|cifar100)_k(?P<k>\d+)_p(?P<p>\d+)_"
                    r"(?P<aug>aug|noaug)_(?P<opt>adam|sgd)_s(?P<seed>\d+)$")


def load_runs(runs_dir, arch, dataset, opt):
    """Return {(k, noise %): DataFrame averaged over seeds} for full-train-set augmented runs."""
    found = {}
    for log in sorted(Path(runs_dir).glob("*/log.csv")):
        m = RUN_RE.match(log.parent.name)
        if not m or (m["arch"], m["dataset"], m["opt"], m["aug"]) != (arch, dataset, opt, "aug"):
            continue
        df = pd.read_csv(log)
        if len(df):
            found.setdefault((int(m["k"]), int(m["p"])), []).append(df)
    return {key: pd.concat(dfs).groupby("epoch", as_index=False).mean(numeric_only=True)
            for key, dfs in sorted(found.items())}


def ref_curve(arch, dataset, noise, k):
    """Return (x, test_err, train_err) from the paper's data, or None if unavailable."""
    folder, entry = REFS.get((arch, dataset), {}).get(noise, (None, None))
    path = REF_DIR / str(folder) / "curves.npz"
    if folder is None or not path.exists():
        return None
    z = np.load(path)
    hit = np.flatnonzero(z["ks"] == k)
    if not len(hit):
        return None
    te, tr = z[f"{entry}__Test Error"][hit[0]], z[f"{entry}__Train Error"][hit[0]]
    return np.arange(1, len(te) + 1), te, tr


def save(fig, stem):
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def plot_widths(a):
    """Fig. 9 left: several widths at one noise level, test + train error (+ Fig. 9 right heatmap)."""
    ours = {k: df for (k, p), df in load_runs(a.runs, a.arch, a.dataset, a.optimizer).items() if p == a.noise}
    widths = sorted(set(ours) | set(a.extra_ref_ks))
    if not widths:
        print(f"widths: no runs for {a.arch}/{a.dataset}/{a.optimizer} at {a.noise}% noise, skipped")
        return
    fig, (ax_te, ax_tr) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for i, k in enumerate(widths):
        color = COLORS[i % len(COLORS)]
        ref = ref_curve(a.arch, a.dataset, a.noise, k)
        if ref is not None:
            ax_te.plot(ref[0], ref[1], "--", color=color, lw=1, alpha=0.6, label=f"k={k} (paper)")
            ax_tr.plot(ref[0], ref[2], "--", color=color, lw=1, alpha=0.6)
        if k in ours:
            df = ours[k]
            ax_te.plot(df["epoch"], df["test_err"], color=color, lw=2, label=f"k={k} (ours)")
            tr = df.dropna(subset=["full_train_err"])
            ax_tr.plot(tr["epoch"], tr["full_train_err"], color=color, lw=2)
    ax_te.set(xscale="log", xlabel="Epoch", ylabel="Test error (clean test set)",
              title=f"Epoch-wise DD: {a.arch}, {a.dataset}, {a.noise}% label noise")
    ax_tr.set(xscale="log", xlabel="Epoch", ylabel="Train error (noisy labels)", title="Train error")
    for ax in (ax_te, ax_tr):
        ax.grid(alpha=0.3)
    ax_te.legend(fontsize=8)
    stem = a.out / f"epochwise_{a.arch}_{a.dataset}_p{a.noise}"
    save(fig, stem)

    folder, entry = REFS.get((a.arch, a.dataset), {}).get(a.noise, (None, None))
    if folder is not None and (REF_DIR / folder / "curves.npz").exists():
        z = np.load(REF_DIR / folder / "curves.npz")
        ks, te = z["ks"], z[f"{entry}__Test Error"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        mesh = ax.pcolormesh(np.arange(1, te.shape[1] + 1), ks, te, shading="auto", cmap="viridis")
        for k in ours:
            ax.axhline(k, color="white", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel="Epoch", ylabel="Width k",
               title=f"Paper data (Fig. 9 right): test error, {a.noise}% noise\n(dotted = our widths)")
        fig.colorbar(mesh, ax=ax, label="Test error")
        save(fig, Path(f"{stem}_heatmap"))


def plot_fig10(a):
    """Fig. 10: for one large model per setting, test (top) and train (bottom) error vs. epochs
    at 0/10/20 % label noise, as in the published figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col")
    for col, (arch, dataset, opt, k, title) in enumerate(FIG10_PANELS):
        ax_te, ax_tr = axes[0, col], axes[1, col]
        ours = load_runs(a.runs, arch, dataset, opt)
        for i, p in enumerate(NOISES):
            color = COLORS[i]
            ref = ref_curve(arch, dataset, p, k)
            if ref is not None:
                ax_te.plot(ref[0], ref[1], "--", color=color, lw=1, alpha=0.6, label=f"{p}% noise (paper)")
                ax_tr.plot(ref[0], ref[2], "--", color=color, lw=1, alpha=0.6)
            if (k, p) in ours:
                df = ours[(k, p)]
                ax_te.plot(df["epoch"], df["test_err"], color=color, lw=2, label=f"{p}% noise (ours)")
                tr = df.dropna(subset=["full_train_err"])
                ax_tr.plot(tr["epoch"], tr["full_train_err"], color=color, lw=2)
        ax_te.set(xscale="log", ylabel="Test error", title=f"{title}, k={k}")
        ax_tr.set(xscale="log", xlabel="Epoch", ylabel="Train error (noisy labels)")
        for ax in (ax_te, ax_tr):
            ax.grid(alpha=0.3)
        ax_te.legend(fontsize=8)
    save(fig, a.out / "fig10_epochwise_noise")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["widths", "fig10", "all"], default="all")
    p.add_argument("--noise", type=int, default=20, help="widths mode: label noise in percent")
    p.add_argument("--arch", default="resnet18k", help="widths mode")
    p.add_argument("--dataset", default="cifar10", help="widths mode")
    p.add_argument("--optimizer", default="adam", help="widths mode")
    p.add_argument("--extra-ref-ks", type=int, nargs="*", default=[], help="paper widths to show w/o our run")
    p.add_argument("--runs", type=Path, default=HERE / "runs")
    p.add_argument("--out", type=Path, default=HERE / "figures")
    a = p.parse_args()
    if a.mode in ("widths", "all"):
        plot_widths(a)
    if a.mode in ("fig10", "all"):
        plot_fig10(a)


if __name__ == "__main__":
    main()
