"""Run the Part D experiment grids, several runs at a time on every visible GPU (or the CPU)."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = ["E5", "E1", "E1n", "E2", "E3", "E1b"]
WIDTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 64]
WIDTHS_E3 = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]
WD_E2 = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 0.0]


def run(exp, arch, k, noise, wd, opt, lr, sched, epochs, seed=0, noise_seed=0, extra=()):
    name = f"{arch}_k{k}_p{noise:.2f}_wd{wd:.0e}_{opt}_s{seed}_n{noise_seed}"
    args = ["--arch", arch, "--k", str(k), "--noise", str(noise), "--wd", str(wd), "--opt", opt,
            "--lr", str(lr), "--lr-sched", sched, "--epochs", str(epochs), "--seed", str(seed),
            "--noise-seed", str(noise_seed), *extra]
    return {"exp": exp, "name": name, "args": args, "arch": arch, "k": k, "epochs": epochs}


def tasks_for(exp):
    T = []
    if exp == "E1":
        for wd in (0.0, 5e-4, 1e-4):
            for k in WIDTHS:
                T.append(run("E1", "cnn5", k, 0.10, wd, "sgd", 0.1, "invsqrt", 1280))
        for seed in (1, 2):
            for wd in (0.0, 5e-4):
                for k in (16, 24, 32, 40):
                    T.append(run("E1", "cnn5", k, 0.10, wd, "sgd", 0.1, "invsqrt", 1280, seed=seed))
    elif exp == "E2":
        for wd in WD_E2:
            T.append(run("E2", "resnet18", 64, 0.20, wd, "sgd", 0.1, "invsqrt", 500))
    elif exp == "E3":
        for member in range(5):
            for k in WIDTHS_E3:
                T.append(run("E3", "resnet18", k, 0.15, 0.0, "adam", 1e-4, "const", 500, seed=member, noise_seed=member))
    elif exp == "E5":
        for p in (0.0, 0.05, 0.10):
            for k in WIDTHS:
                T.append(run("E5", "resnet18", k, p, 0.0, "adam", 1e-4, "const", 500))
    elif exp == "E1n":
        for k in WIDTHS:
            T.append(run("E1n", "cnn5", k, 0.10, 0.0, "sgd", 0.1, "invsqrt", 1459, extra=["--test-noise", "0.1"]))
    elif exp == "E1b":
        for wd in (0.0, 5e-4):
            for k in WIDTHS:
                T.append(run("E1b", "resnet18", k, 0.20, wd, "adam", 1e-4, "const", 500))
    else:
        raise ValueError(exp)
    return T


def cost(t):
    k = t["k"]
    if t["arch"] == "resnet18":
        return 1 if k <= 12 else 2 if k <= 24 else 3 if k <= 40 else 4
    return 1 if k <= 32 else 2


def run_dir(results, t):
    return results / t["exp"] / t["name"]


def devices():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exps", nargs="+", choices=EXPS, default=EXPS)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--capacity", type=int, default=8, help="load units per device; a run costs 1-4 units (cost())")
    p.add_argument("--only-k", type=int, nargs="*", default=None, help="only these widths, e.g. to split the grid over jobs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tasks = [t for e in args.exps for t in tasks_for(e)]
    if args.only_k:
        tasks = [t for t in tasks if t["k"] in args.only_k]
    todo = sorted((t for t in tasks if not (run_dir(args.results, t) / "summary.json").exists()), key=lambda t: -cost(t))
    print(f"{len(tasks)} runs in {' '.join(args.exps)}, {len(todo)} not finished", flush=True)
    if args.dry_run:
        for t in todo:
            print(f"  {t['exp']:4s} {t['name']}")
        return

    sys.path.insert(0, str(HERE))
    from common import prepare_cifar10
    prepare_cifar10(HERE / "data")

    devs = devices()
    load = {g: 0 for g in devs}
    running = {}
    log_dir = args.results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    print(f"devices {devs}, capacity {args.capacity} per device", flush=True)
    while todo or running:
        for proc in list(running):
            if proc.poll() is not None:
                t, g = running.pop(proc)
                load[g] -= cost(t)
                failed += proc.returncode != 0
                print(f"[{time.strftime('%H:%M:%S')}] {'done' if proc.returncode == 0 else 'FAILED'} "
                      f"{t['exp']} {t['name']}", flush=True)
        for t in list(todo):
            g = min(devs, key=load.get)
            if load[g] > 0 and load[g] + cost(t) > args.capacity:
                continue
            todo.remove(t)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            cmd = [sys.executable, str(HERE / "train.py"), *t["args"], "--out", str(run_dir(args.results, t))]
            if g == "cpu":
                cmd += ["--device", "cpu"]
            else:
                env["CUDA_VISIBLE_DEVICES"] = g
            with open(log_dir / f"{t['exp']}_{t['name']}.log", "a") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running[proc] = (t, g)
            load[g] += cost(t)
            print(f"[{time.strftime('%H:%M:%S')}] start {t['exp']} {t['name']} on {g}", flush=True)
        time.sleep(10)
    print(f"finished, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
