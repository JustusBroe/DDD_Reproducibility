"""Shared Part D code: GPU-resident CIFAR-10, label noise, batched augmentation and evaluation."""

import os
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = 10


def prepare_cifar10(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "cifar10.npz"
    lock = data_dir / "cifar10.lock"
    while not out.exists():
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 900:
                lock.unlink(missing_ok=True)
            time.sleep(5)
            continue
        try:
            os.close(fd)
            tar_path = data_dir / "cifar-10-python.tar.gz"
            if not tar_path.exists():
                urllib.request.urlretrieve(CIFAR10_URL, tar_path)
            xs, ys = {}, {}
            with tarfile.open(tar_path) as tar:
                for split, names in (("train", [f"data_batch_{i}" for i in range(1, 6)]), ("test", ["test_batch"])):
                    xb, yb = [], []
                    for name in names:
                        d = pickle.load(tar.extractfile(f"cifar-10-batches-py/{name}"), encoding="latin1")
                        xb.append(np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
                        yb.append(np.asarray(d["labels"], dtype=np.int64))
                    xs[split], ys[split] = np.concatenate(xb), np.concatenate(yb)
            tmp = data_dir / "cifar10.tmp.npz"
            np.savez(tmp, x_train=xs["train"], y_train=ys["train"], x_test=xs["test"], y_test=ys["test"])
            os.replace(tmp, out)
        finally:
            lock.unlink(missing_ok=True)
    return out


def label_noise(y: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y)) < p
    offset = rng.integers(1, NUM_CLASSES, size=len(y))
    return np.where(flip, (y + offset) % NUM_CLASSES, y).astype(np.int64)


class GPUData:
    def __init__(self, data_dir, device, noise=0.0, noise_seed=0, subset=None, test_noise=0.0, test_noise_seed=1000):
        d = np.load(prepare_cifar10(Path(data_dir)))
        x_tr, y_tr, x_te, y_te = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
        if subset:
            x_tr, y_tr = x_tr[:subset], y_tr[:subset]
            x_te, y_te = x_te[: max(subset // 5, 500)], y_te[: max(subset // 5, 500)]
        y_noisy = label_noise(y_tr, noise, noise_seed)
        self.y_test_noisy = (torch.from_numpy(label_noise(y_te, test_noise, test_noise_seed)).to(device)
                             if test_noise > 0 else None)
        self.device = device
        self.x_train = torch.from_numpy(x_tr).to(device)
        self.y_train = torch.from_numpy(y_noisy).to(device)
        self.y_train_clean = torch.from_numpy(y_tr).to(device)
        self.is_noisy = self.y_train != self.y_train_clean
        self.x_test = torch.from_numpy(x_te).to(device)
        self.y_test = torch.from_numpy(y_te).to(device)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.n_train = len(self.y_train)
        self.noise_rate = float(self.is_noisy.float().mean())

    def normalise(self, x_uint8):
        return (x_uint8.float().div_(255.0) - self.mean) / self.std

    def augment(self, x_uint8, gen):
        b = x_uint8.shape[0]
        x = F.pad(x_uint8, (4, 4, 4, 4))
        dy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
        dx = torch.randint(0, 9, (b,), device=x.device, generator=gen)
        ar = torch.arange(32, device=x.device)
        rows = (dy[:, None] + ar)[:, None, :, None]
        cols = (dx[:, None] + ar)[:, None, None, :]
        bidx = torch.arange(b, device=x.device)[:, None, None, None]
        cidx = torch.arange(3, device=x.device)[None, :, None, None]
        x = x[bidx, cidx, rows, cols]
        flip = torch.rand(b, device=x.device, generator=gen) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x

    def train_batches(self, batch_size, gen, aug=True):
        perm = torch.randperm(self.n_train, device=self.device, generator=gen)
        for i in range(0, self.n_train, batch_size):
            idx = perm[i : i + batch_size]
            x = self.x_train[idx]
            if aug:
                x = self.augment(x, gen)
            yield self.normalise(x), self.y_train[idx]


def make_model(arch: str, k: int, num_classes: int = NUM_CLASSES):
    if arch == "resnet18":
        from resnet18k import make_resnet18k
        return make_resnet18k(k=k, num_classes=num_classes)
    if arch == "cnn5":
        from cnn5 import make_cnn
        return make_cnn(c=k, num_classes=num_classes)
    raise ValueError(arch)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, data: GPUData, x_uint8, y, amp_dtype=None, batch_size=1000, return_logits=False, y_noisy=None):
    model.eval()
    loss_sum, err_sum, logits_all = 0.0, 0, []
    loss_n, err_n = 0.0, 0
    for i in range(0, len(y), batch_size):
        x = data.normalise(x_uint8[i : i + batch_size])
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        out = out.float()
        yy = y[i : i + batch_size]
        loss_sum += F.cross_entropy(out, yy, reduction="sum").item()
        err_sum += (out.argmax(1) != yy).sum().item()
        if y_noisy is not None:
            yn = y_noisy[i : i + batch_size]
            loss_n += F.cross_entropy(out, yn, reduction="sum").item()
            err_n += (out.argmax(1) != yn).sum().item()
        if return_logits:
            logits_all.append(out.cpu())
    model.train()
    res = {"loss": loss_sum / len(y), "err": err_sum / len(y)}
    if y_noisy is not None:
        res.update(loss_noisy=loss_n / len(y), err_noisy=err_n / len(y))
    if return_logits:
        res["logits"] = torch.cat(logits_all).numpy()
    return res


@torch.no_grad()
def evaluate_train_full(model, data: GPUData, amp_dtype=None):
    model.eval()
    preds = []
    for i in range(0, data.n_train, 1000):
        x = data.normalise(data.x_train[i : i + 1000])
        with torch.autocast(device_type=x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            preds.append(model(x).float().argmax(1))
    model.train()
    pred = torch.cat(preds)
    noisy = data.is_noisy
    res = {"train_err_full": float((pred != data.y_train).float().mean())}
    if noisy.any():
        res["noisy_fit"] = float((pred[noisy] == data.y_train[noisy]).float().mean())
        res["clean_err"] = float((pred[~noisy] != data.y_train[~noisy]).float().mean())
    return res


def sq_weight_norm(model) -> float:
    return float(sum(p.detach().float().pow(2).sum() for p in model.parameters()))
