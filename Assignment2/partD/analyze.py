"""Aggregate the Part D results, compute double-descent metrics and draw the report figures."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
AUTH = HERE / "authors_data"

PURPLE, BLUE, GOLD = "#5b3a99", "#1f78b4", "#a67c00"
WD_COLORS = {0.0: PURPLE, 1e-4: BLUE, 5e-4: GOLD}
SEQ = LinearSegmentedColormap.from_list("purples", ["#f4f0fa", "#b9a6dc", "#5b3a99", "#21123f"])
_CMAP_FILE = HERE / "colormap_inferno_strong_1.txt"
PAPER_CMAP = (matplotlib.colors.ListedColormap(np.loadtxt(_CMAP_FILE) / 255.0, name="inferno_strong")
              if _CMAP_FILE.exists() else plt.get_cmap("inferno"))
INK, MUTED = "#222222", "#6b6b6b"
PEAK_MIN = 0.015

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.edgecolor": MUTED,
    "lines.linewidth": 1.4, "lines.markersize": 3, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.labelcolor": INK, "text.color": INK,
})


def load_runs(results: Path, exp: str, partial=False):
    rows = []
    for d in sorted((results / exp).glob("*")):
        cfg_f, sum_f, met_f = d / "config.json", d / "summary.json", d / "metrics.csv"
        if not cfg_f.exists() or not met_f.exists():
            continue
        cfg = json.loads(cfg_f.read_text())
        met = pd.read_csv(met_f)
        if sum_f.exists():
            s = json.loads(sum_f.read_text())
            row = {**s, "finished": True}
        elif partial and len(met):
            last = met.iloc[-1]
            full = met.dropna(subset=["train_err_full"])
            row = {"test_err": last.test_err, "test_loss": last.test_loss,
                   "train_err_full": full.train_err_full.iloc[-1] if len(full) else last.train_err,
                   "epochs_done": int(last.epoch), "finished": False}
        else:
            continue
        row.update({"run": d.name, "dir": str(d), "k": int(cfg["k"]), "wd": float(cfg["wd"]), "seed": int(cfg["seed"]),
                    "noise_seed": int(cfg["noise_seed"]), "noise": float(cfg["noise"]), "arch": cfg["arch"],
                    "best_test_err": met.test_err.min(), "epochs_total": int(cfg["epochs"])})
        rows.append(row)
    return pd.DataFrame(rows)


def authors(name):
    f = AUTH / f"{name}.npz"
    return dict(np.load(f)) if f.exists() else None


def dd_metrics(ks, test_err, train_err, eps=(0.1, 0.01)):
    ks, te, tr = map(np.asarray, (ks, test_err, train_err))
    o = np.argsort(ks)
    ks, te, tr = ks[o], te[o], tr[o]
    out = {}
    for e in eps:
        below = np.where(tr <= e)[0]
        out[f"k_interp_{e:g}"] = int(ks[below[0]]) if len(below) else np.nan
    cand = np.where(tr <= 0.5 * tr[0])[0]
    cand = cand if len(cand) else np.arange(len(ks))
    rise = te - np.minimum.accumulate(te)
    i_peak = int(cand[np.argmax(rise[cand])])
    i_min = int(np.argmin(te[: i_peak + 1]))
    out["has_peak"] = bool(rise[i_peak] >= PEAK_MIN)
    after = np.arange(i_peak, len(ks))
    i_min2 = int(after[np.argmin(te[after])])
    out["k_min2"], out["err_min2"] = ks[i_min2].item(), float(te[i_min2])
    out.update({"k_min": ks[i_min].item(), "err_min": float(te[i_min]), "k_peak": ks[i_peak].item(),
                "err_peak": float(te[i_peak]), "peak_height": float(te[i_peak] - te[i_min]),
                "err_largest": float(te[-1]), "k_largest": ks[-1].item()})
    return out


def curve(df, value, by="k"):
    g = df.groupby(by)[value]
    return g.mean().index.values, g.mean().values, g.std().fillna(0).values, g.count().values


def noisy_test_error(err_clean, p, classes=10):
    return 1.0 - (1 - p) * (1 - err_clean) - err_clean * p / (classes - 1)


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def fig_d1(df, out, metrics, df_n=None):
    if df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True)
    auth = {0.0: ("pct-cifar10-mcnn-mdd-50k-p10-sgd-aug-fast", 0), 1e-4: ("pct-cifar10-mcnn-p10-sgd-aug-decay-big", 0),
            5e-4: ("pct-cifar10-mcnn-p10-sgd-aug-decay5-big", 0)}
    for wd, sub in sorted(df.groupby("wd"), key=lambda x: x[0]):
        c = WD_COLORS.get(wd, MUTED)
        lab = "no weight decay" if wd == 0 else f"λ = {wd:.0e}".replace("e-0", "e-")
        s0 = sub[sub.seed == 0]
        noisy = wd == 0 and df_n is not None and "test_err_noisy" in df_n
        src, cols = ((df_n, ("test_err_noisy", "train_err_full", "test_loss_noisy")) if noisy
                     else (s0, ("test_err", "train_err_full", "test_loss")))
        for ax, col in zip(axes, cols):
            k, m, _, _ = curve(src, col)
            ax.plot(k, m, "-o", color=c, label="λ = 0 (noisy test labels)" if noisy else lab, ms=2.5)
            if noisy:
                continue
            ks_, _, sd, n = curve(sub, col)
            rep = n > 1
            if rep.any():
                ax.errorbar(ks_[rep], m[np.isin(k, ks_[rep])], yerr=sd[rep], fmt="none", ecolor=c, elinewidth=0.8,
                            capsize=1.5)
        if wd in (0.0, 5e-4):
            k, m, _, _ = curve(src, "best_test_err")
            axes[0].plot(k, m, ":", color=c, lw=0.9)
        a = authors(auth[wd][0]) if wd in auth else None
        if a is not None:
            t = auth[wd][1]
            sel = a["ks"] <= df.k.max()
            for ax, key in zip(axes, ("test_error", "train_error", "test_loss")):
                ax.plot(a["ks"][sel], a[key][t, sel, -1], "--", color=c, lw=0.9, alpha=0.75)
        k, m, _, _ = curve(s0, "train_err_full")
        k2, te, _, _ = curve(s0, "test_err")
        metrics.append({"exp": "E1 (Fig. 21)", "setting": lab, **dd_metrics(k2, te, m)})
    axes[0].set_ylabel("Test error")
    axes[1].set_ylabel("Train error\n(noisy labels)")
    axes[2].set_ylabel("Test loss")
    axes[2].set_xlabel("CNN width parameter k")
    axes[0].plot([], [], "--", color=MUTED, lw=0.9, label="Nakkiran et al. (raw data)")
    axes[0].plot([], [], ":", color=MUTED, lw=0.9, label="ours, optimal early stopping")
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, loc="upper right")
    axes[0].set_title("5-layer CNN, CIFAR-10, 10% noise, SGD")
    save(fig, "fig21_weight_decay_cnn", out)


def fig_d2(df, out, metrics, results):
    if df.empty:
        return
    df = df.sort_values("wd", ascending=False)
    mats, losses = [], []
    n_ep = int(df.epochs_total.max())
    for _, r in df.iterrows():
        met = pd.read_csv(Path(r["dir"]) / "metrics.csv")
        v = np.full(n_ep, np.nan)
        lo = np.full(n_ep, np.nan)
        v[: len(met)] = met.test_err.values[:n_ep]
        lo[: len(met)] = met.train_err.values[:n_ep]
        mats.append(v)
        losses.append(lo)
    mats, losses = np.array(mats).T, np.array(losses).T

    def inv_label(w):
        if w == 0:
            return "inf"
        v = 1 / w
        if v < 1000:
            return f"{v:.0f}"
        e = int(np.floor(np.log10(v)))
        return f"{v / 10 ** e:.2g}e{e}"
    labels = [inv_label(w) for w in df.wd]
    fig = plt.figure(figsize=(7.0, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.7)
    for j, (M, title, cb) in enumerate(((mats, "Test error", "test error"), (losses, "Train error", "train error"))):
        ax = fig.add_subplot(gs[0, j])
        ep = np.arange(1, n_ep + 1)
        if j == 0:
            norm = matplotlib.colors.LogNorm(vmin=np.nanmin(M), vmax=np.nanmax(M))
        else:
            norm = matplotlib.colors.Normalize(vmin=np.nanmin(M), vmax=0.8, clip=True)
        im = ax.pcolormesh(np.arange(len(labels) + 1), np.r_[ep, n_ep + 1] - 0.5, M, cmap=PAPER_CMAP, shading="flat",
                           norm=norm, rasterized=True)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_xlabel("1/λ")
        ax.set_ylabel("Epoch")
        ax.set_title(title)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.ax.tick_params(labelsize=6)
        if j == 0:
            ticks = [t for t in (0.15, 0.2, 0.3, 0.4, 0.6, 0.8) if norm.vmin <= t <= norm.vmax]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:g}" for t in ticks])
            cbar.ax.minorticks_off()
        else:
            ticks = [round(float(norm.vmin), 2), 0.2, 0.4, 0.6, 0.8]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(labels))
    ax.plot(x, df.test_err.values, "-o", color=PURPLE, label="test error")
    ax.plot(x, df.train_err_full.values, "--o", color=GOLD, label="train error", ms=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_xlabel("1/λ  (less regularisation →)")
    ax.set_ylabel("Error at final epoch")
    ax.legend()
    ax.set_title("Final epoch")
    fig.suptitle("ResNet18 (k = 64), CIFAR-10, 20% noise, SGD: weight-decay sweep", y=1.04, fontsize=8.5)
    inv = np.array([np.inf if w == 0 else 1 / w for w in df.wd])
    m = dd_metrics(np.arange(len(inv)), df.test_err.values, df.train_err_full.values)
    m.update({key: f"1/λ={labels[m[key]]}" for key in ("k_min", "k_peak", "k_min2")})
    for e in (0.1, 0.01):
        i = m[f"k_interp_{e:g}"]
        m[f"k_interp_{e:g}"] = f"1/λ={labels[i]}" if i == i else "-"
    m["k_largest"] = "1/λ=∞"
    metrics.append({"exp": "E2 (Fig. 22)", "setting": "k=64, λ sweep", **m})
    save(fig, "fig22_weight_decay_sweep", out)


def ensemble_curve(df):
    rows = []
    for k, sub in df.groupby("k"):
        logits = [np.load(Path(d) / "test_logits.npz") for d in sub.dir if (Path(d) / "test_logits.npz").exists()]
        if len(logits) < 2:
            continue
        y = logits[0]["y_test"]
        L = np.stack([l["logits"].astype(np.float32) for l in logits])
        votes = np.zeros(L.shape[1:])
        np.add.at(votes, (np.arange(L.shape[1])[None, :].repeat(len(L), 0), L.argmax(2)), 1.0)
        p = np.exp(L - L.max(2, keepdims=True))
        p = (p / p.sum(2, keepdims=True)).mean(0)
        pred = (votes + 1e-3 * p).argmax(1)
        rows.append({"k": k, "members": len(L), "ens_err": float((pred != y).mean()),
                     "single_mean": float(sub.test_err.mean()), "single_std": float(sub.test_err.std())})
    return pd.DataFrame(rows)


def fig_d3(df, out, metrics):
    if df.empty:
        return
    ens = ensemble_curve(df[df.finished]) if "finished" in df else ensemble_curve(df)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    k, m, s, n = curve(df, "test_err")
    ax.plot(k, m, "-o", color=PURPLE, label=f"single model (mean ± sd, n≤{n.max()})")
    ax.fill_between(k, m - s, m + s, color=PURPLE, alpha=0.2, lw=0)
    if len(ens):
        ax.plot(ens.k, ens.ens_err, "-o", color=GOLD, label=f"ensemble of {ens.members.max()} (plurality vote)")
    a = authors("cifar10-resnet18k-p15-adam-reps")
    if a is not None:
        ax.plot(a["ks"], a["test_error"][:, :, -1].mean(0), "--", color=PURPLE, lw=0.9, alpha=0.75,
                label="Nakkiran et al. single (4K ep.)")
    ax.set_ylabel("Test error")
    ax.legend(loc="upper right")
    ax.set_title("ResNet18, CIFAR-10, 15% noise, Adam")
    ax = axes[1]
    kt, tr, _, _ = curve(df, "train_err_full")
    ax.plot(kt, tr, "--o", color=PURPLE, ms=2.5, label="train error (single)")
    if len(ens):
        ax.plot(ens.k, ens.single_mean - ens.ens_err, "-o", color=GOLD, ms=2.5, label="ensemble gain Δ(k)")
    ax.set_xlabel("ResNet18 width parameter k")
    ax.set_ylabel("Error")
    ax.legend(loc="upper right")
    metrics.append({"exp": "E3 (Fig. 28)", "setting": "single model", **dd_metrics(k, m, tr)})
    if len(ens):
        e = ens.set_index("k")
        metrics.append({"exp": "E3 (Fig. 28)", "setting": "ensemble",
                        **dd_metrics(e.index.values, e.ens_err.values, np.interp(e.index.values, kt, tr)),
                        "max_gain": float((e.single_mean - e.ens_err).max()),
                        "k_max_gain": int((e.single_mean - e.ens_err).idxmax())})
        ens.to_csv(out.parent / "tables" / "ensemble.csv", index=False)
    save(fig, "fig28_ensembling", out)


def fig_d4(df, out, metrics):
    if df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=True)
    for wd, sub in sorted(df.groupby("wd"), key=lambda x: x[0]):
        c = WD_COLORS.get(wd, MUTED)
        lab = "no weight decay" if wd == 0 else f"λ = {wd:.0e}".replace("e-0", "e-")
        for ax, col in zip(axes, ("test_err", "train_err_full")):
            k, m, s, n = curve(sub, col)
            ax.plot(k, m, "-o", color=c, label=lab, ms=2.5)
        k, tr, _, _ = curve(sub, "train_err_full")
        k2, te, _, _ = curve(sub, "test_err")
        metrics.append({"exp": "E1b (ext.)", "setting": lab, **dd_metrics(k2, te, tr)})
    a = authors("cifar10-resnet18k-p20-decay")
    if a is not None:
        for t, c in ((0, PURPLE), (1, GOLD)):
            axes[0].plot(a["ks"], a["test_error"][t, :, -1], "--", color=c, lw=0.9, alpha=0.75)
        axes[0].plot([], [], "--", color=MUTED, lw=0.9, label="Nakkiran et al. (4K ep., λ unknown)")
    axes[0].set_ylabel("Test error")
    axes[1].set_ylabel("Train error")
    axes[1].set_xlabel("ResNet18 width parameter k")
    axes[0].legend(loc="upper right")
    axes[0].set_title("ResNet18, CIFAR-10, 20% noise, Adam")
    save(fig, "figExt_resnet_adam_weight_decay", out)


def tex(label):
    return label.replace("λ", r"$\lambda$").replace("∞", r"$\infty$").replace("≤", r"$\le$")


def noise_curves(dfs):
    cur = {}
    e5, e3, e1b = dfs.get("E5"), dfs.get("E3"), dfs.get("E1b")
    if e5 is not None and len(e5):
        for p, sub in e5.groupby("noise"):
            cur[round(p, 2)] = sub
    if e3 is not None and len(e3):
        cur[0.15] = e3[(e3.seed == 0) & (e3.noise_seed == 0)]
    if e1b is not None and len(e1b):
        cur[0.20] = e1b[e1b.wd == 0]
    out = {}
    for p, sub in sorted(cur.items()):
        k, te, sd, n = curve(sub, "test_err")
        _, tr, _, _ = curve(sub, "train_err_full")
        out[p] = (k, te, sd, tr)
    return out


def authors_noise_curves():
    out = {}
    a = authors("cifar10-resnet18k-50k-adam")
    if a is not None:
        for t, p in enumerate((0.0, 0.10, 0.20)):
            out[p] = (a["ks"], a["test_error"][t, :, -1], a["train_error"][t, :, -1])
    a = authors("pct-cf10-res18-50k-p05-adam")
    if a is not None:
        out[0.05] = (a["ks"], a["test_error"][0, :, -1], a["train_error"][0, :, -1])
    a = authors("cifar10-resnet18k-p15-adam-reps")
    if a is not None:
        out[0.15] = (a["ks"], a["test_error"][:, :, -1].mean(0), a["train_error"][:, :, -1].mean(0))
    return dict(sorted(out.items()))


def fig_d5(dfs, out, metrics):
    ours, auth = noise_curves(dfs), authors_noise_curves()
    if len(ours) < 2:
        return
    cmap = plt.get_cmap("viridis")
    col = {p: cmap(v) for p, v in zip((0.0, 0.05, 0.10, 0.15, 0.20), (0.0, 0.25, 0.47, 0.66, 0.84))}
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.95), gridspec_kw={"width_ratios": [1.25, 1.0, 0.85, 0.85], "wspace": 0.45})
    rows = []
    for src, curves in (("ours, 500 ep.", ours), ("paper, 4K ep.", auth)):
        for p, c in curves.items():
            k, te, tr = c[0], c[1], c[-1]
            if src.startswith("ours"):
                axes[0].plot(k, te, "-o", color=col[p], ms=1.8, lw=1.1, label=f"{p:.0%}")
                axes[1].plot(k, tr, "-o", color=col[p], ms=1.8, lw=1.1)
                if (c[2] > 0).any():
                    axes[0].fill_between(k, te - c[2], te + c[2], color=col[p], alpha=0.2, lw=0)
            else:
                axes[0].plot(k, te, "--", color=col[p], lw=0.7, alpha=0.8)
            m = dd_metrics(k, te, tr)
            rows.append({"source": src, "noise": p, **m})
            metrics.append({"exp": "D3 noise", "setting": f"{src}, p={p:.2f}", **m})
    nm = pd.DataFrame(rows)
    nm.to_csv(out.parent / "tables" / "noise_metrics.csv", index=False)
    axes[0].set_xlabel("ResNet18 width k")
    axes[0].set_ylabel("Test error")
    axes[0].set_title("(a) test error (dashed: paper)")
    axes[0].legend(title="label noise", fontsize=5.5, title_fontsize=6, loc="upper right", handlelength=1.2)
    axes[1].set_xlabel("ResNet18 width k")
    axes[1].set_ylabel("Train error (noisy labels)")
    axes[1].set_title("(b) train error (ours)")
    for src, mk, ls in (("ours, 500 ep.", "o", "-"), ("paper, 4K ep.", "s", "--")):
        s = nm[nm.source == src].sort_values("noise")
        x = 100 * s.noise.values
        axes[2].plot(x, s.peak_height.values, ls, marker=mk, color=PURPLE if mk == "o" else MUTED, ms=3, label=src)
        kp = np.where(s.has_peak.values, s.k_peak.values.astype(float), np.nan)
        axes[3].plot(x, kp, ls, marker=mk, color=PURPLE if mk == "o" else MUTED, ms=3)
        axes[3].plot(x, s["k_interp_0.1"].values, ls, marker=mk, color=GOLD if mk == "o" else "#c9b27a", ms=3, mfc="white")
    axes[2].set_xlabel("Label noise (%)")
    axes[2].set_ylabel("Peak height h")
    axes[2].set_title("(c) peak height")
    axes[2].legend(fontsize=5.5, loc="upper left")
    axes[3].set_xlabel("Label noise (%)")
    axes[3].set_ylabel("Width k")
    axes[3].set_title("(d) $k^\\star$ (filled), $k_{0.1}$ (open)")
    for ax in axes[2:]:
        ax.set_xticks([0, 5, 10, 15, 20])
    save(fig, "fig4b_label_noise", out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--out", type=Path, default=HERE / "figures")
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    (args.out.parent / "tables").mkdir(parents=True, exist_ok=True)
    metrics = []
    dfs = {e: load_runs(args.results, e, args.partial) for e in ("E1", "E2", "E3", "E1b", "E5", "E1n")}
    for e, df in dfs.items():
        if len(df):
            print(f"{e}: {len(df)} runs ({int(df.finished.sum())} finished)")
            df.drop(columns=["dir"]).to_csv(args.out.parent / "tables" / f"runs_{e}.csv", index=False)
    fig_d1(dfs["E1"], args.out, metrics, dfs["E1n"])
    fig_d2(dfs["E2"], args.out, metrics, args.results)
    fig_d3(dfs["E3"], args.out, metrics)
    fig_d4(dfs["E1b"], args.out, metrics)
    fig_d5(dfs, args.out, metrics)
    if metrics:
        mt = pd.DataFrame(metrics)
        mt.to_csv(args.out.parent / "tables" / "metrics.csv", index=False)
        cols = ["exp", "setting", "k_interp_0.1", "k_peak", "err_peak", "peak_height", "err_largest"]
        mt_tex = mt.copy()
        for c in ("setting", "k_min", "k_peak", "k_interp_0.1", "k_interp_0.01"):
            if c in mt_tex:
                mt_tex[c] = mt_tex[c].map(lambda v: tex(v) if isinstance(v, str) else v)
        mt_tex[[c for c in cols if c in mt_tex]].to_latex(args.out.parent / "tables" / "metrics.tex", index=False,
                                                  float_format="%.3f", na_rep="-", escape=False)
        print(mt.to_string())


if __name__ == "__main__":
    main()
