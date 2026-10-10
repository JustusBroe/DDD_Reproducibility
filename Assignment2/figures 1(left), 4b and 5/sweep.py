"""Resumable (and shardable) width sweep. One CSV row per finished run, plus a curves CSV.

Scaled-down CNN example (Fig. 5a, no augmentation):
  python sweep.py --arch cnn --opt sgd --steps 40000 --noises 0 0.1 0.2 --augs 0 \
      --ks 1 2 3 4 5 6 8 10 12 14 16 20 24 32 --seeds 0 --out results/cnn_sgd_noaug.csv

Cluster: launch the same command with --shard i/N (i=0..N-1); each shard writes its own file.
Re-running a command skips every (arch, opt, steps, aug, noise, k, seed) already in the CSV.
"""
import argparse
import csv
import os
import sys
from itertools import product

from data import get_device, load_cifar
from train import train_run

KEY = ["arch", "opt", "steps", "aug", "noise", "k", "seed"]
FIELDS = KEY + ["params", "train_err", "train_loss", "test_err", "test_loss", "seconds", "device"]
CURVE_FIELDS = KEY + ["step", "epoch", "train_err", "train_loss", "test_err", "test_loss"]


def key_of(row):
    return (row["arch"], row["opt"], int(row["steps"]), int(row["aug"]),
            float(row["noise"]), int(row["k"]), int(row["seed"]))


def read_done(*paths):
    done = set()
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                done |= {key_of(r) for r in csv.DictReader(f)}
    return done


def append_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn", choices=["cnn", "resnet"])
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--steps", type=int, default=40000, help="gradient steps (batch 128; 391 steps/epoch)")
    ap.add_argument("--epochs", type=int, default=None, help="alternative to --steps (epochs*391)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 32])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    ap.add_argument("--augs", type=int, nargs="+", default=[0], help="0 = no augmentation, 1 = crop+flip")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_eval", type=int, default=30, help="number of log-spaced evaluations per run")
    ap.add_argument("--shard", default="0/1", help="i/N: run only jobs with index %% N == i")
    ap.add_argument("--out", default="results/run.csv")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--save_dir", default="models",
                    help="folder for final weights and test logits of every run ('' = do not save)")
    ap.add_argument("--noise_seed", type=int, default=None,
                    help="seed of the label-noise draw; default = the run's seed. A fixed value (e.g. 0) makes "
                         "all seeds share one noisy training set, so they differ only in initialisation")
    ap.add_argument("--ckpt_dir", default="ckpt",
                    help="folder for resume checkpoints of unfinished runs (only used with --ckpt_every)")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="checkpoint every N epochs so a killed run resumes (0 = off; use ~10 for long runs)")
    a = ap.parse_args()

    steps = a.epochs * 391 if a.epochs else a.steps
    si, sn = map(int, a.shard.split("/"))
    out = a.out if sn == 1 else a.out.replace(".csv", f"_shard{si}of{sn}.csv")
    curves_out = out.replace(".csv", "_curves.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # skip anything already finished by any shard of this sweep
    base = a.out.replace(".csv", "")
    siblings = [a.out] + [f"{base}_shard{i}of{sn}.csv" for i in range(sn)]
    done = read_done(*siblings)

    device = get_device()
    data = load_cifar(a.dataset, a.data_root, device)
    jobs = [j for j in product(a.seeds, a.augs, a.noises, a.ks)]
    jobs = [j for idx, j in enumerate(jobs) if idx % sn == si]
    print(f"device={device} steps={steps} jobs(this shard)={len(jobs)} already done={len(done)}", flush=True)

    for seed, aug, noise, k in jobs:
        if (a.arch, a.opt, steps, aug, noise, k, seed) in done:
            continue
        final, curve = train_run(a.arch, k, noise, bool(aug), seed, a.opt, steps, data, device,
                                 n_eval=a.n_eval, log=lambda s: print(s, flush=True),
                                 save_dir=a.save_dir or None, noise_seed=a.noise_seed,
                                 ckpt_dir=a.ckpt_dir or None, ckpt_every=a.ckpt_every)
        key = dict(arch=a.arch, opt=a.opt, steps=steps, aug=aug, noise=noise, k=k, seed=seed)
        row = {**key, **{f: final[f] for f in ["params", "train_err", "train_loss", "test_err",
                                              "test_loss", "seconds"]}, "device": str(device)}
        append_rows(curves_out, CURVE_FIELDS, [{**key, **c} for c in curve])
        append_rows(out, FIELDS, [row])                      # written last => row means "run finished"
        print(f"DONE {key} train_err={final['train_err']:.4f} test_err={final['test_err']:.4f} "
              f"params={final['params']} ({final['seconds']}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
