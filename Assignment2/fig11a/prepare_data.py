"""Download CIFAR-10 and draw one nested noisy training subset.

perm is a permutation of the 50k training indices. The subset of size n is
perm[:n], so 12500 is contained in 25000 is contained in 50000. Label noise is
applied once to the full training set (probability NOISE, uniform over the
other 9 classes) and then inherited by every subset, which is the paper's
"sampled once and not per epoch" rule. Augmentations of a sample therefore
share that fixed noisy label.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))


def apply_noise(labels, prob, rng):
    noisy = labels.copy()
    flip = rng.random(len(labels)) < prob
    offset = rng.integers(1, 10, size=len(labels))
    noisy[flip] = (labels[flip] + offset[flip]) % 10
    return noisy, flip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "cifar-data")))
    p.add_argument("--noise", type=float, default=float(os.environ.get("NOISE", "0.1")))
    p.add_argument("--subset-seed", type=int, default=int(os.environ.get("SUBSET_SEED", "0")))
    p.add_argument("--noise-seed", type=int, default=int(os.environ.get("NOISE_SEED", "0")))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out = args.data_dir / "subsets.npz"
    if out.exists() and not args.force:
        saved = np.load(out)
        same = (
            abs(float(saved["noise"][0]) - args.noise) < 1e-8
            and int(saved["subset_seed"][0]) == args.subset_seed
            and int(saved["noise_seed"][0]) == args.noise_seed
        )
        if not same:
            raise SystemExit(
                f"{out} was built with different noise/seeds. Delete it or pass --force."
            )
        print(f"reusing {out} ({int(saved['n_flipped'][0])} / {len(saved['perm'])} labels flipped)")
        return

    train = CIFAR10(args.data_dir / "cifar10", train=True, download=True)
    # Also fetch the test split now so batch jobs do not all download it later.
    CIFAR10(args.data_dir / "cifar10", train=False, download=True)
    labels = np.asarray(train.targets, dtype=np.int64)
    if len(labels) != 50000:
        raise SystemExit(f"expected 50000 CIFAR-10 train images, got {len(labels)}")

    perm = np.random.default_rng(args.subset_seed).permutation(len(labels))
    noisy, flip = apply_noise(labels, args.noise, np.random.default_rng(args.noise_seed))
    if flip.any() and np.any(noisy[flip] == labels[flip]):
        raise SystemExit("label noise produced a correct label")

    np.savez(
        str(out),
        perm=perm.astype(np.int64),
        noisy_labels=noisy.astype(np.int64),
        clean_labels=labels,
        noise=np.array([args.noise]),
        subset_seed=np.array([args.subset_seed]),
        noise_seed=np.array([args.noise_seed]),
        n_flipped=np.array([int(flip.sum())]),
    )
    print(
        f"wrote {out}: flipped {int(flip.sum())}/{len(labels)} "
        f"({100.0 * flip.mean():.2f}%) with noise={args.noise}, "
        f"subset_seed={args.subset_seed}, noise_seed={args.noise_seed}"
    )


if __name__ == "__main__":
    main()
