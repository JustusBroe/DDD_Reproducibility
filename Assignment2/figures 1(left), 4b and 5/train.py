"""Training loop for width-swept CNN / ResNet18 (Nakkiran et al., 2021, App. B.2).

Two optimizer modes, both batch size 128, no weight decay, default PyTorch init:
  sgd : plain SGD (no momentum), lr(t) = 0.1 / sqrt(1 + floor(t / 512)) per gradient step
  adam: Adam, constant lr 1e-4
Training length is counted in gradient steps (for Adam: epochs * steps_per_epoch).

Long runs can be checkpointed (ckpt_dir + ckpt_every): every `ckpt_every` epochs the model, optimizer,
step counter, training curve and random-number state are written to <ckpt_dir>/<tag>.ckpt (atomic
write). If a run is killed (preemption, node failure) and started again with the same arguments, it
resumes from the last checkpoint instead of step 0. The checkpoint is deleted when the run finishes.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import add_label_noise, get_device, load_cifar, make_augmenter
from mcnn import make_cnn
from resnet18k import make_resnet18k

BATCH = 128


def build_model(arch, k, num_classes):
    if arch == "cnn":
        return make_cnn(k, num_classes)
    if arch == "resnet":
        return make_resnet18k(k, num_classes)
    raise ValueError(arch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def sgd_lr(t, lr0=0.1, L=512):
    return lr0 / math.sqrt(1 + t // L)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Returns (error, mean cross-entropy) in eval mode (BN uses running stats)."""
    model.eval()
    wrong, loss = 0, 0.0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        wrong += (out.argmax(1) != y[i:i + bs]).sum()
        loss += nn.functional.cross_entropy(out, y[i:i + bs], reduction="sum")
    model.train()
    return wrong.item() / len(x), loss.item() / len(x)


def eval_steps(total, n_points=30):
    """Log-spaced evaluation steps (always includes the final step)."""
    pts = np.unique(np.round(np.geomspace(50, total, n_points)).astype(int))
    return set(pts.tolist()) | {total}


def run_tag(arch, opt, total_steps, aug, noise, k, seed):
    """Unique name of one run; used for saved models and checkpoints."""
    return f"{arch}_{opt}_s{total_steps}_aug{int(aug)}_p{noise}_k{k}_seed{seed}"


def _save_ckpt(path, model, optim, step, curve, elapsed, device):
    """Atomic write: a run killed in the middle of saving leaves the previous checkpoint intact."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = dict(
        model=model.state_dict(), optim=optim.state_dict(), step=step, curve=curve, elapsed=elapsed,
        rng_cpu=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state() if device.type == "cuda" else None,
    )
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def train_run(arch, k, noise, aug, seed, opt, total_steps, data, device,
              n_eval=30, log=None, save_dir=None, ckpt_dir=None, ckpt_every=0, crash_at=None,
              noise_seed=None):
    """Train one model and return (final_metrics_dict, curve_list).

    `seed` sets the network initialisation and the data order; `noise_seed` sets which training labels
    are corrupted (None = same as `seed`). Passing a fixed noise_seed makes trials differ only in their
    random initialisation, as in the paper's Fig. 1 ("five trials, with random network initialization").

    If `save_dir` is given, the final weights (<tag>.pt, a state_dict) and the model's outputs on the
    clean test set (<tag>_test_logits.pt, shape [10000, C]) are saved there before returning.
    If `ckpt_dir` and `ckpt_every` (epochs) are given, the run is checkpointed and resumes automatically.
    `crash_at` (a step number) hard-kills the process at that step; it exists only for test_resume.py.
    """
    xtr, ytr, xte, yte, C = data
    torch.manual_seed(seed)
    y_noisy = add_label_noise(ytr, noise, seed if noise_seed is None else noise_seed, C)   # fixed once
    model = build_model(arch, k, C).to(device)
    params = n_params(model)
    if opt == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    augment = make_augmenter(device) if aug else None
    loss_fn = nn.CrossEntropyLoss()
    checkpoints = eval_steps(total_steps, n_eval)
    n = len(xtr)
    spe = math.ceil(n / BATCH)                               # gradient steps per epoch
    tag = run_tag(arch, opt, total_steps, aug, noise, k, seed)
    ckpt = os.path.join(ckpt_dir, tag + ".ckpt") if (ckpt_dir and ckpt_every) else None

    curve, step, elapsed0 = [], 0, 0.0
    if ckpt and os.path.exists(ckpt):
        # load on the CPU: load_state_dict copies tensors to the GPU itself, and Adam's step counters
        # must stay on the CPU (on the GPU every optimizer step would force a slow synchronisation)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        step, curve, elapsed0 = state["step"], state["curve"], state["elapsed"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        if log:
            log(f"  [{tag}] resumed from checkpoint at step {step}/{total_steps}")
    t0 = time.time()
    model.train()
    while step < total_steps:
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            if step >= total_steps:
                break
            idx = perm[i:i + BATCH]
            xb, yb = xtr[idx], y_noisy[idx]
            if augment is not None:
                xb = augment(xb)
            if opt == "sgd":
                for g in optim.param_groups:
                    g["lr"] = sgd_lr(step)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            step += 1
            if crash_at is not None and step == crash_at:    # test hook: simulate a killed job
                os._exit(3)
            if step in checkpoints:
                tr_err, tr_loss = evaluate(model, xtr, y_noisy, 1000)   # noisy labels, no aug
                te_err, te_loss = evaluate(model, xte, yte, 1000)       # clean labels
                curve.append(dict(step=step, epoch=step * BATCH / n, train_err=tr_err,
                                  train_loss=tr_loss, test_err=te_err, test_loss=te_loss))
                if log:
                    log(f"  [{arch} k={k} p={noise} aug={int(aug)} s={seed}] step {step}/{total_steps} "
                        f"train_err={tr_err:.4f} test_err={te_err:.4f} "
                        f"({elapsed0 + time.time() - t0:.0f}s)")
        else:
            # the epoch ended normally (no break): checkpoint at the epoch boundary
            if ckpt and step < total_steps and (step // spe) % ckpt_every == 0:
                _save_ckpt(ckpt, model, optim, step, curve, elapsed0 + time.time() - t0, device)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, tag + ".pt"))
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(xte[i:i + 1000]) for i in range(0, len(xte), 1000)])
        torch.save(logits.cpu(), os.path.join(save_dir, tag + "_test_logits.pt"))
        model.train()
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)                                      # run finished: the checkpoint is not needed
    final = dict(curve[-1])
    final.update(params=params, seconds=round(elapsed0 + time.time() - t0, 1))
    return final, curve


def benchmark(arch="cnn", ks=(4, 16, 32), steps=200, aug=False, fake=False, data_root="./data"):
    """Time `steps` SGD steps for a few widths to size the sweep budget."""
    device = get_device()
    data = load_cifar("cifar10", data_root, device=device, fake=fake)
    print(f"device: {device}")
    for k in ks:
        model = build_model(arch, k, 10).to(device)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        augment = make_augmenter(device) if aug else None
        xtr, ytr = data[0], data[1]
        for s in range(steps + 20):
            if s == 20:                      # 20 warm-up steps excluded from timing
                _sync(device)
                t0 = time.time()
            idx = torch.randint(0, len(xtr), (BATCH,), device=device)
            xb = augment(xtr[idx]) if augment else xtr[idx]
            optim.zero_grad()
            nn.functional.cross_entropy(model(xb), ytr[idx]).backward()
            optim.step()
        _sync(device)
        dt = (time.time() - t0) / steps
        print(f"{arch} k={k:3d}: {dt * 1000:6.1f} ms/step -> {dt * 50_000:7.0f} s per 50k steps, "
              f"{dt * 500_000 / 3600:5.1f} h per 500k steps")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Time a few training steps per width.")
    ap.add_argument("--arch", default="cnn", choices=["cnn", "resnet"])
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 32])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--fake", action="store_true", help="random data, no download")
    ap.add_argument("--data_root", default="./data", help="folder containing cifar-10-batches-py/")
    a = ap.parse_args()
    benchmark(a.arch, a.ks, a.steps, a.aug, a.fake, a.data_root)
