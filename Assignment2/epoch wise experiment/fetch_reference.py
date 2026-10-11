"""Download the authors' raw training curves and convert them to .npz.

Source: public GCS bucket gs://hml-public/dd/<experiment>/{Mlist,ks}, linked
from https://gitlab.com/harvard-machine-learning/double-descent. `Mlist` is a
pickled list of dicts (one per noise level / repetition); each value has shape
[len(ks), n_epochs], i.e. the full per-epoch curve for every width. These
are the curves behind the paper's epoch-wise figures.

Unpickling untrusted data can execute code, so a restricted unpickler that only
allows numpy arrays and plain containers is used.

    python fetch_reference.py                              # default ResNet18 CIFAR-10 set
    python fetch_reference.py cifar10-resnet18k-50k-adam-all
"""
import io
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/hml-public/dd"
DEFAULT = [
    "cifar10-resnet18k-50k-adam",       # Fig. 2, 9, 10a (entries: 0/10/20 % noise)
    "cifar100-resnet18k-50k-adam",      # Fig. 10b
    # Fig. 10c (CNN) has no matching public file: its epoch-wise data is in the authors' private bucket.
]

ALLOWED = {
    (m, n)
    for m in ("numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric", "numpy._core.numeric")
    for n in ("_reconstruct", "scalar", "_frombuffer")
} | {("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "OrderedDict")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}; inspect before allowing")


def safe_load(raw):
    return SafeUnpickler(io.BytesIO(raw)).load()


def fetch(name, root):
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    raw = {}
    for f in ("Mlist", "ks"):
        path = out / f
        if not path.exists():
            print(f"downloading {BASE}/{name}/{f}")
            urllib.request.urlretrieve(f"{BASE}/{name}/{f}", path)
        raw[f] = path.read_bytes()
    Ms, ks = safe_load(raw["Mlist"]), np.asarray(safe_load(raw["ks"]))
    if isinstance(Ms, dict):
        Ms = [Ms]

    arrays, summary = {"ks": ks}, []
    for i, M in enumerate(Ms):
        info = {}
        for key, v in M.items():
            v = np.asarray(v)
            arrays[f"{i}__{key}"] = v
            info[key] = list(v.shape)
        te = np.asarray(M.get("Test Error", []))
        if te.ndim == 2:
            info["final_test_err_at_max_k"] = float(te[-1, -1])
            info["min_test_err_at_max_k"] = float(te[-1].min())
        summary.append(info)
    np.savez_compressed(out / "curves.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(dict(ks=ks.tolist(), entries=summary), indent=2))

    print(f"\n{name}: {len(Ms)} entries, ks = {ks.tolist()}")
    for i, info in enumerate(summary):
        print(f"  entry {i}: {info}")
    print("  Entries are not labelled with their noise level; higher final test error at max k"
          " => higher noise. Record the mapping you infer in the report.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "reference"
    for name in sys.argv[1:] or DEFAULT:
        fetch(name, root)
