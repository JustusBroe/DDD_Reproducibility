"""Run the (d_model, n) grid of train_one.sh jobs, several jobs per GPU in parallel.

The models are tiny (~1.7M params), so a single run leaves most of a GPU idle;
--jobs-per-gpu runs that many trainings concurrently on each visible GPU.
Finished runs (results/<run>.json) are skipped, unfinished ones resume from their
last checkpoint, so the script can simply be relaunched after a UCloud job ends.
Use --shard/--num-shards to split the grid across several UCloud jobs.
"""

import argparse
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))


def coarse_to_fine(n_values):
    """Order n so the first finished runs already span the whole curve."""
    def level(n):
        for i, step in enumerate((16000, 8000, 4000)):
            if n % step == 0:
                return i
        return 3
    return sorted(n_values, key=lambda n: (level(n), n))


def visible_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 80])
    p.add_argument("--n-min", type=int, default=4000)
    p.add_argument("--n-max", type=int, default=120000)
    p.add_argument("--n-step", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs-per-gpu", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    n_values = coarse_to_fine(range(args.n_min, args.n_max + 1, args.n_step))
    tasks = [(d, n) for n in n_values for d in args.sizes]
    tasks = tasks[args.shard :: args.num_shards]

    result_dir = PROJECT_ROOT / "results"
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [(d, n) for d, n in tasks if not (result_dir / f"d{d}_n{n}_s{args.seed}.json").exists()]
    print(f"shard {args.shard}/{args.num_shards}: {len(tasks)} runs, {len(todo)} still to do")

    if args.dry_run:
        for d, n in todo:
            print(f"  d={d} n={n}")
        return

    gpus = visible_gpus()
    if not gpus:
        sys.exit("no GPU visible")
    slots = queue.Queue()
    for _ in range(args.jobs_per_gpu):
        for g in gpus:
            slots.put(g)
    print(f"GPUs {gpus}, {args.jobs_per_gpu} jobs each -> {slots.qsize()} parallel runs")

    def run(task):
        d, n = task
        gpu = slots.get()
        name = f"d{d}_n{n}_s{args.seed}"
        start = time.time()
        try:
            with open(log_dir / f"{name}.log", "a") as log:
                rc = subprocess.call(
                    ["bash", str(SCRIPT_DIR / "train_one.sh"), str(d), str(n), str(args.seed)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                )
        finally:
            slots.put(gpu)
        status = "ok" if rc == 0 else f"FAILED (rc={rc}, see logs/{name}.log)"
        print(f"[{time.strftime('%H:%M:%S')}] {name} on GPU {gpu}: {status} "
              f"after {(time.time() - start) / 60:.1f} min", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        rcs = list(pool.map(run, todo))
    failed = sum(rc != 0 for rc in rcs)
    print(f"done: {len(rcs) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
