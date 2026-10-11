"""Train one ResNet18-k or 5-layer CNN-k on CIFAR-10 with label noise and weight decay."""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import GPUData, count_params, evaluate, evaluate_train_full, make_model, sq_weight_norm

FIELDS = ["epoch", "step", "lr", "train_loss", "train_err", "test_loss", "test_err", "wnorm2",
          "train_err_full", "noisy_fit", "clean_err", "time_s"]
FIELDS_TEST_NOISE = ["test_loss_noisy", "test_err_noisy"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["resnet18", "cnn5"], default="resnet18")
    p.add_argument("--k", type=int, required=True, help="width parameter")
    p.add_argument("--noise", type=float, default=0.0, help="label noise probability p")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--test-noise", type=float, default=0.0,
                   help="also score the test set against labels with this much noise (logged as *_noisy)")
    p.add_argument("--test-noise-seed", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="init / data-order seed")
    p.add_argument("--wd", type=float, default=0.0, help="L2 weight decay (coupled, torch.optim default)")
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-sched", choices=["const", "invsqrt"], default="const",
                   help="invsqrt: lr(t) = lr / sqrt(1 + floor(t / lr-interval)), t = gradient step")
    p.add_argument("--lr-interval", type=int, default=512)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--no-aug", dest="aug", action="store_false")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--subset", type=int, default=None, help="use only the first N train samples (smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.add_argument("--full-train-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly (checkpointed) after this long")
    return p.parse_args(argv)


def lr_at(args, step):
    if args.lr_sched == "invsqrt":
        return args.lr / math.sqrt(1 + step // args.lr_interval)
    return args.lr


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print(f"{out} already finished")
        return
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1))

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if (args.amp == "bf16" and use_cuda) else None
    mem_fmt = torch.channels_last if (args.channels_last and use_cuda) else torch.contiguous_format

    data = GPUData(args.data_dir, device, noise=args.noise, noise_seed=args.noise_seed, subset=args.subset,
                   test_noise=args.test_noise, test_noise_seed=args.test_noise_seed)
    fields = FIELDS + (FIELDS_TEST_NOISE if data.y_test_noisy is not None else [])

    torch.manual_seed(args.seed)
    model = make_model(args.arch, args.k).to(device, memory_format=mem_fmt)
    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    gen = torch.Generator(device=device)
    gen.manual_seed(10_000 + args.seed)

    start_epoch, step, elapsed = 0, 0, 0.0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        start_epoch, step, elapsed = ck["epoch"], ck["step"], ck.get("elapsed", 0.0)
        print(f"resumed from epoch {start_epoch}")
    csv_path = out / "metrics.csv"
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_params = count_params(model)
    print(f"{args.arch} k={args.k} params={n_params} noise={args.noise} (actual {data.noise_rate:.4f}) "
          f"wd={args.wd} opt={args.opt} lr={args.lr}/{args.lr_sched} epochs={args.epochs} device={device} amp={amp_dtype}")

    def checkpoint(epoch):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                     "epoch": epoch, "step": step, "elapsed": elapsed}, ckpt_path)

    t_run = time.time()
    diverged = False
    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        err_sum = torch.zeros((), device=device)
        seen = 0
        for x, y in data.train_batches(args.bs, gen, aug=args.aug):
            for g in opt.param_groups:
                g["lr"] = lr_at(args, step)
            x = x.contiguous(memory_format=mem_fmt)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += loss.detach() * len(y)
            err_sum += (logits.detach().argmax(1) != y).sum()
            seen += len(y)
        train_loss = loss_sum.item() / seen
        if not math.isfinite(train_loss):
            diverged = True
        te = evaluate(model, data, data.x_test, data.y_test, amp_dtype, y_noisy=data.y_test_noisy)
        row = {"epoch": epoch, "step": step, "lr": lr_at(args, step), "train_loss": train_loss,
               "train_err": err_sum.item() / seen, "test_loss": te["loss"], "test_err": te["err"],
               "wnorm2": sq_weight_norm(model), "train_err_full": "", "noisy_fit": "", "clean_err": ""}
        if "loss_noisy" in te:
            row.update(test_loss_noisy=te["loss_noisy"], test_err_noisy=te["err_noisy"])
        if epoch % args.full_train_every == 0 or epoch == args.epochs or diverged:
            row.update(evaluate_train_full(model, data, amp_dtype))
        elapsed += time.time() - t0
        row["time_s"] = round(elapsed, 1)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        if epoch % 10 == 0 or epoch == start_epoch + 1:
            print(f"ep {epoch:4d} | train {row['train_loss']:.3f}/{row['train_err']:.3f} "
                  f"| test {row['test_loss']:.3f}/{row['test_err']:.3f} | {time.time() - t0:.1f}s/ep", flush=True)
        if diverged:
            print("non-finite training loss, stopping")
            break
        if epoch % args.ckpt_every == 0:
            checkpoint(epoch)
        if args.max_minutes and (time.time() - t_run) / 60 > args.max_minutes and epoch < args.epochs:
            checkpoint(epoch)
            print(f"time budget reached at epoch {epoch}, checkpointed")
            return

    final = evaluate(model, data, data.x_test, data.y_test, amp_dtype, return_logits=True, y_noisy=data.y_test_noisy)
    extra = {} if data.y_test_noisy is None else {"y_test_noisy": data.y_test_noisy.cpu().numpy()}
    np.savez_compressed(out / "test_logits.npz", logits=final["logits"].astype(np.float16),
                        y_test=data.y_test.cpu().numpy(), **extra)
    summary = {"arch": args.arch, "k": args.k, "noise": args.noise, "noise_seed": args.noise_seed,
               "seed": args.seed, "wd": args.wd, "opt": args.opt, "lr": args.lr, "lr_sched": args.lr_sched,
               "epochs_done": epoch, "steps": step, "params": n_params, "noise_rate_actual": data.noise_rate,
               "test_loss": final["loss"], "test_err": final["err"], "train_time_s": round(elapsed, 1),
               "diverged": diverged, **evaluate_train_full(model, data, amp_dtype)}
    if "loss_noisy" in final:
        summary.update(test_noise=args.test_noise, test_loss_noisy=final["loss_noisy"], test_err_noisy=final["err_noisy"])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    ckpt_path.unlink(missing_ok=True)
    print("done", json.dumps(summary))


if __name__ == "__main__":
    main()
