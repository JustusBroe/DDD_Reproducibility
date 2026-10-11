"""Train one 5-layer CNN (CNN.py) on a noisy CIFAR-10 subset and evaluate it.

Protocol from Nakkiran et al., arXiv:1912.02292, Appendix B.2, for Figure 11(a):
  SGD, momentum 0, weight decay 0, batch size 128, cross-entropy
  lr(t) = lr0 / sqrt(1 + floor(t / 512)) at gradient step t (0-based)
  RandomCrop(32, padding=4), RandomHorizontalFlip; the noisy label is fixed
  per training image and shared by every augmentation of it
  no dropout
  default PyTorch initialization (whatever make_cnn() builds)

Test metrics use the clean CIFAR-10 test set. Train metrics use the unaugmented
training images and the noisy labels the model was fit to, which is what hits
zero at the interpolation threshold. Errors are stored as percentages, same
convention as the transformer results.csv. Loss is mean cross-entropy in nats.

The paper never states input normalization. Images are normalized with the
CIFAR-10 training-set channel statistics used by the Myrtle.ai recipe this
architecture is taken from. Set NORMALIZE=0 to train on [0, 1] pixels instead.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from CNN import make_cnn
from runname import FULL_STEPS, run_name

# Myrtle.ai / standard CIFAR-10 train-set statistics, rounded to 4 decimals.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def expected_params(width):
    """Closed form for make_cnn(width). k=64 is 1,558,026 in the paper."""
    c = int(width)
    return 378 * c * c + 152 * c + 10


def inverse_sqrt_lr(step, lr0, period):
    return lr0 / math.sqrt(1 + step // period)


class ArrayCIFAR(Dataset):
    """Subset of CIFAR images stored as uint8 HWC, transformed per sample."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), int(self.labels[i])


def build_transforms(normalize):
    to_float = [transforms.ToTensor()]
    if normalize:
        to_float.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    eval_tf = transforms.Compose(to_float)
    train_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *([transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)] if normalize else []),
        ]
    )
    return train_tf, eval_tf


def make_loader(dataset, batch_size, shuffle, workers, generator=None):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if shuffle and generator is not None:
        kwargs["generator"] = generator
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, wrong, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        wrong += (logits.argmax(dim=1) != y).sum().item()
        total += y.size(0)
    model.train()
    return {
        "error": 100.0 * wrong / total,
        "loss": loss_sum / total,
        "n": total,
    }


def save_checkpoint(path, model, opt, step, generator, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history,
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=int(os.environ.get("TRAIN_SEED", "0")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", str(FULL_STEPS))))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    p.add_argument("--lr0", type=float, default=float(os.environ.get("LR0", "0.1")))
    p.add_argument("--lr-period", type=int, default=int(os.environ.get("LR_PERIOD", "512")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("NUM_WORKERS", "2")))
    p.add_argument("--save-every", type=int, default=int(os.environ.get("SAVE_EVERY", "5000")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("EVAL_EVERY", "10000")))
    p.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "200")))
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--result-dir", type=Path, default=None)
    p.add_argument("--normalize", type=int, default=int(os.environ.get("NORMALIZE", "1")))
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    args.data_dir = args.data_dir or Path(os.environ.get("DATA_DIR", root / "cifar-data"))
    args.ckpt_dir = args.ckpt_dir or Path(os.environ.get("CKPT_DIR", root / "checkpoints-cnn"))
    args.result_dir = args.result_dir or Path(os.environ.get("RESULT_DIR", root / "results-cnn"))

    name = run_name(args.width, args.n, args.seed, args.max_steps)
    save_dir = args.ckpt_dir / name
    ckpt_path = save_dir / "checkpoint.pt"
    out_path = args.result_dir / f"{name}.json"
    heartbeat = save_dir / "heartbeat"
    save_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[{name}] already evaluated: {out_path}")
        return

    subsets_path = args.data_dir / "subsets.npz"
    if not subsets_path.exists():
        raise SystemExit(f"missing {subsets_path} - run 01_prepare_data.sh")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # V100-era FP32. TF32 would change the SGD trajectory on Blackwell.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("highest")
    elif args.max_steps >= FULL_STEPS:
        print(f"[{name}] WARNING: CUDA is not available; this budget is meant for a GPU", flush=True)

    saved = np.load(subsets_path)
    perm = saved["perm"]
    if args.n > len(perm) or args.n < args.batch_size:
        raise SystemExit(f"n={args.n} is outside [batch size, {len(perm)}]")
    indices = perm[: args.n]
    noisy = saved["noisy_labels"][indices]
    n_flipped = int((noisy != saved["clean_labels"][indices]).sum())

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = make_cnn(c=args.width, num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != expected_params(args.width):
        raise SystemExit(
            f"make_cnn({args.width}) has {n_params} parameters, expected {expected_params(args.width)}"
        )
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr0, momentum=0.0, weight_decay=0.0)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    step = 0
    history = []
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = int(state["step"])
        generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        history = list(state.get("history", []))
        print(f"[{name}] resume at step {step}", flush=True)

    train_base = CIFAR10(args.data_dir / "cifar10", train=True, download=False)
    test_base = CIFAR10(args.data_dir / "cifar10", train=False, download=False)
    train_images = np.ascontiguousarray(np.asarray(train_base.data)[indices])
    test_images = np.ascontiguousarray(np.asarray(test_base.data))
    test_labels = np.asarray(test_base.targets, dtype=np.int64)
    if train_images.shape[1:] != (32, 32, 3):
        raise SystemExit(f"unexpected CIFAR image shape {train_images.shape}")
    del train_base, test_base
    train_tf, eval_tf = build_transforms(bool(args.normalize))
    train_ds = ArrayCIFAR(train_images, noisy, train_tf)
    train_eval_ds = ArrayCIFAR(train_images, noisy, eval_tf)
    test_ds = ArrayCIFAR(test_images, test_labels, eval_tf)
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers, generator)
    # Evaluation is a few dozen batches of in-memory images; extra workers only contend.
    train_eval_loader = make_loader(train_eval_ds, 512, False, 0)
    test_loader = make_loader(test_ds, 512, False, 0)

    print(
        f"[{name}] width={args.width} n={args.n} seed={args.seed} steps={args.max_steps} "
        f"params={n_params} noise_in_subset={n_flipped}/{args.n} "
        f"normalize={bool(args.normalize)} device={device}",
        flush=True,
    )

    def checkpoint():
        save_checkpoint(ckpt_path, model, opt, step, generator, history)
        heartbeat.touch()

    def on_signal(signum, _frame):
        print(f"[{name}] signal {signum}, checkpointing at step {step}", flush=True)
        if step > 0:
            checkpoint()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    heartbeat.touch()

    def run_eval(tag):
        heartbeat.touch()
        train_m = evaluate(model, train_eval_loader, device)
        test_m = evaluate(model, test_loader, device)
        row = {
            "step": step,
            "train_error": train_m["error"],
            "train_loss": train_m["loss"],
            "test_error": test_m["error"],
            "test_loss": test_m["loss"],
        }
        history.append(row)
        print(
            f"[{name}] {tag} step {step}  train_err {train_m['error']:.2f}%  "
            f"test_err {test_m['error']:.2f}%  test_loss {test_m['loss']:.4f}",
            flush=True,
        )
        heartbeat.touch()

    model.train()
    data_iter = iter(train_loader)
    window_start = time.time()
    window_step = step
    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        lr = inverse_sqrt_lr(step, args.lr0, args.lr_period)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        step += 1

        if step % args.log_every == 0 or step == args.max_steps:
            now = time.time()
            rate = (step - window_step) / max(now - window_start, 1e-6)
            print(
                f"[{name}] step {step}/{args.max_steps}  lr {lr:.5f}  "
                f"loss {loss.item():.4f}  {rate:.1f} steps/s",
                flush=True,
            )
            window_start, window_step = now, step
            heartbeat.touch()
        if args.eval_every > 0 and step % args.eval_every == 0 and step < args.max_steps:
            run_eval("mid")
        if step % args.save_every == 0:
            checkpoint()

    checkpoint()
    run_eval("final")
    checkpoint()
    final = history[-1]
    payload = {
        "n": args.n,
        "size": args.width,
        "width": args.width,
        "test_error": final["test_error"],
        "test_loss": final["test_loss"],
        "train_error": final["train_error"],
        "train_loss": final["train_loss"],
        "seed": args.seed,
        "num_updates": step,
        "num_params": n_params,
        "noise": float(saved["noise"][0]),
        "n_flipped": n_flipped,
        "normalize": bool(args.normalize),
        "batch_size": args.batch_size,
        "lr0": args.lr0,
        "lr_period": args.lr_period,
        "history": history,
    }
    write_json(out_path, payload)
    print(f"[{name}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
