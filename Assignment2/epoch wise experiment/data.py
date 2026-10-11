"""CIFAR-10/100 with fixed label noise, kept on the GPU with on-GPU augmentation.

Label noise follows the paper (Sec. 3): with probability p a training label is
replaced by a uniformly random *incorrect* label. The noise is sampled once from
`noise_seed` (not per epoch), so every augmented view of a sample has the same
label and every team member gets the same noisy subset.

Keeping the whole dataset on the device and doing RandomCrop(32, pad=4) +
HorizontalFlip as batched tensor ops avoids the DataLoader bottleneck, which
dominates for small widths over thousands of epochs.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# CIFAR-10 statistics, also used for CIFAR-100 (they differ by < 0.01).
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def make_noisy_labels(labels, p, num_classes, seed):
    """Return (noisy_labels, flipped_mask). Each label flips w.p. p to a uniform wrong class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    flipped = rng.random(len(labels)) < p
    # label + offset (mod C) with offset uniform in 1..C-1 is uniform over the wrong classes.
    offset = rng.integers(1, num_classes, size=len(labels))
    noisy = np.where(flipped, (labels + offset) % num_classes, labels)
    return noisy, flipped


DATASETS = {"cifar10": (torchvision.datasets.CIFAR10, 10), "cifar100": (torchvision.datasets.CIFAR100, 100)}


def load_cifar(name, root, noise, noise_seed, n_train=50000):
    cls, num_classes = DATASETS[name]
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    x_train, y_train = train.data, np.asarray(train.targets)
    if n_train < len(y_train):
        # Fixed subset, seeded by noise_seed so it is shared across widths/seeds.
        idx = np.random.default_rng(noise_seed + 12345).permutation(len(y_train))[:n_train]
        x_train, y_train = x_train[idx], y_train[idx]
    y_noisy, flipped = make_noisy_labels(y_train, noise, num_classes, noise_seed)
    return dict(x_train=x_train, y_train=y_noisy, y_train_clean=y_train, flipped=flipped,
                x_test=test.data, y_test=np.asarray(test.targets), num_classes=num_classes)


def random_crop_flip(x, gen, pad=4):
    """Batched RandomCrop(H, padding=pad) + RandomHorizontalFlip on an (B,C,H,W) tensor."""
    B, C, H, W = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad)).permute(0, 2, 3, 1)  # B,H+2p,W+2p,C
    dy = torch.randint(0, 2 * pad + 1, (B,), generator=gen).to(dev)
    dx = torch.randint(0, 2 * pad + 1, (B,), generator=gen).to(dev)
    rows = dy[:, None] + torch.arange(H, device=dev)
    cols = dx[:, None] + torch.arange(W, device=dev)
    bi = torch.arange(B, device=dev)[:, None, None]
    out = xp[bi, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2)
    flip = (torch.rand(B, generator=gen) < 0.5).to(dev)
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()


class GPULoader:
    """Minimal in-memory loader. Iterating yields normalized (x, y) batches on `device`."""

    def __init__(self, x_uint8, y, batch_size, device, augment=False, shuffle=False, seed=0):
        self.x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.batch_size, self.device = batch_size, device
        self.augment, self.shuffle = augment, shuffle
        self.gen = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, generator=self.gen).to(self.device) if self.shuffle else None
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size] if order is not None else slice(i, i + self.batch_size)
            x = self.x[idx].float().div_(255)
            if self.augment:
                x = random_crop_flip(x, self.gen)
            yield (x - self.mean) / self.std, self.y[idx]
