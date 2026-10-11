"""Epoch-wise double descent (Nakkiran et al. 2019, Sec. 6 / Fig. 2, 9, 10).

Trains one model (one width k, one noise level, one seed) and logs train/test
loss and error after EVERY epoch to <out>/log.csv. Defaults match the paper's
ResNet18 setup: Adam, constant lr 1e-4, batch 128, cross-entropy, default init,
no weight decay, RandomCrop+Flip augmentation, 4K epochs.

`--optimizer sgd` is the paper's CNN setup (App. B.2): plain SGD (no momentum)
with lr(t) = 0.1 / sqrt(1 + floor(t / 512)) at gradient step t, for 500K steps
(= 1279 epochs of 391 steps on 50k samples).

Safe to kill and re-run with the same arguments: it resumes from <out>/last.pt.
Re-running with a larger --epochs extends a finished run.

Examples:
    python train.py --k 12 --noise 0.2 --epochs 1000                       # Fig. 2/9
    python train.py --dataset cifar100 --k 64 --noise 0.1                  # Fig. 10b
    python train.py --arch mcnn --optimizer sgd --k 128 --noise 0.2 --epochs 1279   # Fig. 10c
"""
import math
import argparse
import contextlib
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from CNN import make_cnn
from data import DATASETS, GPULoader, load_cifar
from resnet18k import make_resnet18k

FIELDS = ["epoch", "step", "lr", "train_loss", "train_err", "test_loss", "test_err",
          "full_train_err", "noisy_fit", "clean_train_err", "epoch_time"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["resnet18k", "mcnn"], default="resnet18k")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    p.add_argument("--k", type=int, required=True, help="width parameter")
    p.add_argument("--noise", type=float, default=0.2, help="label-noise probability p")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-4 (adam), 0.1 (sgd)")
    p.add_argument("--lr-step", type=int, default=512, help="sgd: inverse-sqrt decay every N steps")
    p.add_argument("--seed", type=int, default=0, help="init + batch-order seed")
    p.add_argument("--noise-seed", type=int, default=0, help="keep fixed across the team")
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA (faster; a deviation, report it)")
    p.add_argument("--eval-train-every", type=int, default=10,
                   help="full pass over the train set (eval mode, no aug) every N epochs; 0 = off")
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--save-weights-at", type=int, nargs="*", default=[],
                   help="extra epochs at which to keep a permanent weight snapshot")
    p.add_argument("--data-dir", default=os.environ.get("CIFAR_DIR", "./data"))
    p.add_argument("--out-root", default="./runs")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(a):
    aug = "noaug" if a.no_aug else "aug"
    n = "" if a.n_train == 50000 else f"_n{a.n_train}"
    return f"{a.arch}_{a.dataset}_k{a.k}_p{round(a.noise * 100)}_{aug}_{a.optimizer}{n}_s{a.seed}"


@torch.no_grad()
def predict(model, loader, autocast):
    """Return (sum of CE loss, predictions) over a loader, in eval mode."""
    model.eval()
    loss, preds = torch.zeros((), device=loader.device), []
    for x, y in loader:
        with autocast():
            out = model(x)
        loss += F.cross_entropy(out.float(), y, reduction="sum")
        preds.append(out.argmax(1))
    return loss.item(), torch.cat(preds)


def main():
    a = parse_args()
    device = pick_device(a.device)
    out = Path(a.out_root) / run_name(a)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    use_amp = a.amp and device.type == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

    if a.lr is None:
        a.lr = 1e-4 if a.optimizer == "adam" else 0.1
    d = load_cifar(a.dataset, a.data_dir, a.noise, a.noise_seed, a.n_train)
    train_loader = GPULoader(d["x_train"], d["y_train"], a.batch_size, device,
                             augment=not a.no_aug, shuffle=True, seed=a.seed)
    train_eval_loader = GPULoader(d["x_train"], d["y_train"], 1000, device)
    test_loader = GPULoader(d["x_test"], d["y_test"], 1000, device)
    flipped = torch.as_tensor(d["flipped"], device=device)
    y_noisy, y_clean = train_eval_loader.y, torch.as_tensor(d["y_train_clean"], device=device)

    torch.manual_seed(a.seed)
    C = d["num_classes"]
    model = (make_resnet18k(a.k, C) if a.arch == "resnet18k" else make_cnn(a.k, C)).to(device)
    if a.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        lr_at = lambda step: 1.0
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.0)
        lr_at = lambda step: 1.0 / math.sqrt(1 + step // a.lr_step)
    # Stepped once per gradient step; the scheduler state is restored with the checkpoint.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    n_params = sum(p.numel() for p in model.parameters())

    start_epoch, ckpt_path, log_path = 0, out / "last.pt", out / "log.csv"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        # map_location moves every tensor to the device, but RNG states must be CPU ByteTensors.
        train_loader.gen.set_state(ck["gen"].cpu())
        torch.set_rng_state(ck["torch_rng"].cpu())
        start_epoch = ck["epoch"]
        print(f"resumed {out.name} from epoch {start_epoch}")
    # Drop log rows written after the checkpoint so a resumed run has no duplicates.
    rows = []
    if log_path.exists():
        with open(log_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= start_epoch]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)

    if start_epoch == 0:
        cfg = vars(a) | dict(device=str(device), n_params=n_params, run=out.name,
                             actual_noise_frac=float(d["flipped"].mean()))
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        np.save(out / "flipped_idx.npy", np.flatnonzero(d["flipped"]))
    print(f"{out.name}: {n_params / 1e6:.2f}M params on {device}, "
          f"{d['flipped'].mean():.3f} of train labels flipped")

    for epoch in range(start_epoch + 1, a.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = torch.zeros((), device=device)
        wrong = torch.zeros((), device=device)
        for x, y in train_loader:
            with autocast():
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(y)
            wrong += (logits.argmax(1) != y).sum()
        n = len(train_loader.y)
        row = dict(epoch=epoch, step=sched.last_epoch, lr=opt.param_groups[0]["lr"],
                   train_loss=loss_sum.item() / n, train_err=wrong.item() / n)

        test_loss, test_pred = predict(model, test_loader, autocast)
        row.update(test_loss=test_loss / len(test_loader.y),
                   test_err=(test_pred != test_loader.y).float().mean().item())

        if a.eval_train_every and (epoch % a.eval_train_every == 0 or epoch == a.epochs):
            _, tr_pred = predict(model, train_eval_loader, autocast)
            row.update(full_train_err=(tr_pred != y_noisy).float().mean().item(),
                       # fraction of mislabeled samples the net has memorised (predicts the noisy label)
                       noisy_fit=(tr_pred[flipped] == y_noisy[flipped]).float().mean().item()
                       if flipped.any() else float("nan"),
                       clean_train_err=(tr_pred != y_clean).float().mean().item())
        row["epoch_time"] = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        if epoch % 10 == 0 or epoch == start_epoch + 1:
            print(f"ep {epoch:5d} | train {row['train_loss']:.3f} / {row['train_err']:.3f} | "
                  f"test {row['test_loss']:.3f} / {row['test_err']:.3f} | {row['epoch_time']:.1f}s", flush=True)

        if epoch % a.ckpt_every == 0 or epoch == a.epochs:
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                      gen=train_loader.gen.get_state(), torch_rng=torch.get_rng_state())
            torch.save(ck, out / "last.tmp")
            os.replace(out / "last.tmp", ckpt_path)
        if epoch in a.save_weights_at:
            torch.save(model.state_dict(), out / f"weights_ep{epoch}.pt")

    (out / "DONE").touch()


if __name__ == "__main__":
    main()
