"""Download and decode the authors' public raw results (gs://hml-public/dd) with a restricted unpickler."""

import argparse
import io
import pickle
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://storage.googleapis.com/hml-public/dd"
OUT_DIR = Path(__file__).resolve().parent / "authors_data"

EXPERIMENTS = {
    "cifar10-mcnn-p10-sgd": "Fig. 11a: CNN, 10% noise, SGD, 3 training-set sizes",
    "pct-cifar10-mcnn-mdd-50k-p10-sgd-aug-fast": "Fig. 21 (lambda=0) / 27: CNN, 10% noise, aug, SGD, no weight decay",
    "pct-cf10-res18-50k-p05-adam": "Fig. 4b: ResNet18, 5% noise, Adam",
    "pct-cifar10-mcnn-p10-sgd-aug-decay-big": "Fig. 21: CNN, 10% noise, aug, SGD, wd=1e-4",
    "pct-cifar10-mcnn-p10-sgd-aug-decay5-big": "Fig. 21: CNN, 10% noise, aug, SGD, wd=5e-4",
    "cifar10-resnet18k-p20-decay": "ResNet18, 20% noise, weight decay",
    "pct-cifar10-resnet18-50k-p20-adam-decay": "ResNet18, 20% noise, Adam, weight decay",
    "cifar10-resnet18k-p15-adam-reps": "Fig. 1/28: ResNet18, 15% noise, Adam, 5 trials",
    "cifar10-resnet18k-50k-adam": "Fig. 4b: ResNet18, 0/10/20% noise, Adam",
}

ALLOWED = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        module = module.replace("numpy.core", "numpy._core") if hasattr(np, "_core") else module
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)


def safe_load(raw: bytes):
    return NumpyOnlyUnpickler(io.BytesIO(raw), encoding="latin1").load()


def fetch(name: str, fname: str, raw_dir: Path) -> bytes:
    path = raw_dir / name / fname
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{name}/{fname}", path)
    return path.read_bytes()


def to_arrays(M):
    trials = M if isinstance(M, (list, tuple)) else [M]
    out = {}
    for key in ("Test Error", "Test Loss", "Train Error", "Train Loss"):
        out[key.lower().replace(" ", "_")] = np.stack([np.asarray(t[key], dtype=np.float32) for t in trials])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="*", default=list(EXPERIMENTS))
    p.add_argument("--raw-dir", type=Path, default=OUT_DIR / "raw")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        ks = np.atleast_1d(np.asarray(safe_load(fetch(name, "ks", args.raw_dir))))
        arrays = to_arrays(safe_load(fetch(name, "Mlist", args.raw_dir)))
        np.savez_compressed(OUT_DIR / f"{name}.npz", ks=ks, **arrays)
        shape = arrays["test_error"].shape
        print(f"{name}: ks[{len(ks)}]={ks.min()}..{ks.max()}  arrays [trials, sizes, epochs]={shape}"
              f"  -- {EXPERIMENTS.get(name, '')}")


if __name__ == "__main__":
    main()
