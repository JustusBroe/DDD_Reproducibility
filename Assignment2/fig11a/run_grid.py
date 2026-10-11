"""Run the (width, n) grid of train_one.sh jobs, several jobs per GPU in parallel.

Finished runs (results-cnn/<run>.json) are skipped. Unfinished ones resume from
their last checkpoint, so this can be relaunched after a UCloud job hits its
time limit. --shard/--num-shards splits the grid across instances that each
have one GPU. Within a shard, wide-stride widths are queued first so a partial
grid already spans the double-descent curve.
"""

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runname import FULL_STEPS, default_ns, default_widths, run_name

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))


def coarse_to_fine(widths):
    def level(k):
        if k % 16 == 0:
            return 0
        if k % 8 == 0:
            return 1
        if k % 4 == 0:
            return 2
        if k % 2 == 0:
            return 3
        return 4

    return sorted(widths, key=lambda k: (level(k), k))


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", type=int, nargs="*", default=None)
    p.add_argument("--n-values", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", "0")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", str(FULL_STEPS))))
    p.add_argument("--jobs-per-gpu", type=int, default=int(os.environ.get("JOBS_PER_GPU", "4")))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.widths:
        widths = args.widths
    elif os.environ.get("WIDTHS"):
        widths = [int(x) for x in os.environ["WIDTHS"].split()]
    else:
        widths = default_widths()
    n_values = args.n_values or (
        [int(x) for x in os.environ["N_VALUES"].split()] if os.environ.get("N_VALUES") else default_ns()
    )
    if not 0 <= args.shard < args.num_shards:
        sys.exit(f"shard must be in 0..{args.num_shards - 1}")

    tasks = [(w, n) for w in coarse_to_fine(widths) for n in n_values]
    tasks = tasks[args.shard :: args.num_shards]
    weight = sum(w * w for w, _ in tasks)

    result_dir = PROJECT_ROOT / "results-cnn"
    log_dir = PROJECT_ROOT / "logs-cnn"
    todo = [
        (w, n) for w, n in tasks
        if not (result_dir / f"{run_name(w, n, args.seed, args.max_steps)}.json").exists()
    ]
    print(
        f"shard {args.shard}/{args.num_shards}: {len(tasks)} runs, {len(todo)} still to do, "
        f"compute weight {weight / 64**2:.2f} x one width-64 run"
    )
    if args.dry_run:
        for w, n in todo:
            print(f"  c={w} n={n}")
        return

    if not todo:
        return
    log_dir.mkdir(parents=True, exist_ok=True)

    gpus = visible_gpus()
    if not gpus:
        sys.exit("no GPU visible")
    slots = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in gpus:
            slots.put(g)
    print(f"GPUs {gpus}, {args.jobs_per_gpu} jobs each -> {slots.qsize()} parallel runs")

    def run(task):
        w, n = task
        gpu = slots.get()
        name = run_name(w, n, args.seed, args.max_steps)
        start = time.time()
        try:
            with open(log_dir / f"{name}.log", "a") as log:
                rc = subprocess.call(
                    ["bash", str(SCRIPT_DIR / "train_one.sh"), str(w), str(n), str(args.seed)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "MAX_STEPS": str(args.max_steps)},
                )
        finally:
            slots.put(gpu)
        status = "ok" if rc == 0 else f"FAILED (rc={rc}, see logs-cnn/{name}.log)"
        print(
            f"[{time.strftime('%H:%M:%S')}] {name} on GPU {gpu}: {status} "
            f"after {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
        return rc

    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        rcs = list(pool.map(run, todo))
    failed = sum(rc != 0 for rc in rcs)
    print(f"done: {len(rcs) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
