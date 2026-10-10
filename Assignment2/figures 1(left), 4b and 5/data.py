"""CIFAR data, fixed-seed label noise and batched GPU augmentation.
"""
import numpy as np
import torch
import os
import pickle



MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize(x_uint8):
    x = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _find_cifar10_folder(root):

    for cand in (os.path.join(root, "cifar-10-batches-py"), root):
        if os.path.exists(os.path.join(cand, "data_batch_1")):
            return cand
    return None


def _load_cifar10_pickles(folder):

    def read(name):
        with open(os.path.join(folder, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"], np.array(d[b"labels"])

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = read("test_batch")
    to_nhwc = lambda a: np.ascontiguousarray(a.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
    return to_nhwc(np.concatenate(xs)), np.concatenate(ys), to_nhwc(xte), yte


def load_cifar(name="cifar10", root="./data", device=None, fake=False, n_fake=2000):
    """Returns (x_train, y_train, x_test, y_test, num_classes), all on `device`.

    CIFAR-10 is read from a local `cifar-10-batches-py/` folder under `root` if present
    (no download, no torchvision needed); otherwise torchvision downloads it into `root`.
    `fake=True` gives random data of the right shape (used for offline tests only).
    """
    device = device or get_device()
    num_classes = 10 if name == "cifar10" else 100
    if fake:
        g = torch.Generator().manual_seed(0)
        xtr = torch.randn(n_fake, 3, 32, 32, generator=g)
        ytr = torch.randint(0, num_classes, (n_fake,), generator=g)
        xte = torch.randn(n_fake // 4, 3, 32, 32, generator=g)
        yte = torch.randint(0, num_classes, (n_fake // 4,), generator=g)
    elif name == "cifar10" and _find_cifar10_folder(root) is not None:
        a, b, c, d = _load_cifar10_pickles(_find_cifar10_folder(root))
        xtr, ytr = _normalize(a), torch.from_numpy(b).long()
        xte, yte = _normalize(c), torch.from_numpy(d).long()
    else:
        from torchvision import datasets
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        tr = cls(root, train=True, download=True)
        te = cls(root, train=False, download=True)
        xtr, ytr = _normalize(tr.data), torch.tensor(tr.targets)
        xte, yte = _normalize(te.data), torch.tensor(te.targets)
    return (xtr.to(device), ytr.to(device), xte.to(device), yte.to(device), num_classes)


def add_label_noise(y, p, seed, num_classes):
    """Replace a fraction p of labels by a uniformly random different class.

    Exactly round(p * n) samples are corrupted; the new label is
    (y + U{1..C-1}) mod C, so it is uniform over the C-1 wrong classes.
    """
    if p <= 0:
        return y.clone()
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.choice(n, size=int(round(p * n)), replace=False)
    shift = rng.randint(1, num_classes, size=len(idx))
    y_np = y.cpu().numpy().copy()
    y_np[idx] = (y_np[idx] + shift) % num_classes
    return torch.from_numpy(y_np).to(y.device)


def make_augmenter(device, pad=4):
    """RandomCrop(32, padding=4) + RandomHorizontalFlip, on a batch, on-device.

    Padding uses black pixels (value 0 before normalization), as torchvision does.
    """
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    pad_val = (0.0 - mean) / std
    ar = torch.arange(32, device=device)

    def augment(x):
        b = x.shape[0]
        xp = pad_val.expand(b, 3, 32 + 2 * pad, 32 + 2 * pad).clone()
        xp[:, :, pad:pad + 32, pad:pad + 32] = x
        i = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        j = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        rows = (i[:, None] + ar)[:, None, :, None]          # B,1,32,1
        cols = (j[:, None] + ar)                              # B,32
        flip = torch.rand(b, device=x.device) < 0.5
        cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]  # B,1,1,32
        bi = torch.arange(b, device=x.device)[:, None, None, None]
        ci = torch.arange(3, device=x.device)[None, :, None, None]
        return xp[bi, ci, rows, cols]

    return augment
