"""Create nested random subsets of the IWSLT'14 de-en train split and binarise them.

Subset n consists of the first n entries of one fixed random permutation of the
train split, so smaller subsets are contained in larger ones. Every subset is
binarised with the dictionaries of the full data (data-bin/full), and the
full valid/test sets are symlinked in, so all runs share vocabulary and test set.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SRC, TGT = "de", "en"


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def binarise_subset(n, perm, src_lines, tgt_lines, full_bin: Path, out_dir: Path, workers: int):
    idx = np.sort(perm[:n])
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "train"
        for lang, lines in ((SRC, src_lines), (TGT, tgt_lines)):
            with open(f"{prefix}.{lang}", "w", encoding="utf-8") as f:
                f.writelines(lines[i] for i in idx)
        subprocess.run(
            [
                "fairseq-preprocess",
                "--source-lang", SRC, "--target-lang", TGT,
                "--trainpref", str(prefix),
                "--srcdict", str(full_bin / f"dict.{SRC}.txt"),
                "--tgtdict", str(full_bin / f"dict.{TGT}.txt"),
                "--destdir", str(out_dir),
                "--workers", str(workers),
            ],
            check=True,
        )
    for split in ("valid", "test"):
        for lang in (SRC, TGT):
            for ext in ("bin", "idx"):
                name = f"{split}.{SRC}-{TGT}.{lang}.{ext}"
                link = out_dir / name
                if not link.exists():
                    os.symlink(os.path.relpath(full_bin / name, out_dir), link)
    np.save(out_dir / "train_indices.npy", idx)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-dir", type=Path, required=True)
    p.add_argument("--full-bin", type=Path, required=True)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--n-min", type=int, default=4000)
    p.add_argument("--n-max", type=int, default=120000)
    p.add_argument("--n-step", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    src_lines = read_lines(args.text_dir / f"train.{SRC}")
    tgt_lines = read_lines(args.text_dir / f"train.{TGT}")
    assert len(src_lines) == len(tgt_lines), "train.de and train.en differ in length"
    pool = len(src_lines)
    print(f"train pool: {pool} sentence pairs")
    assert args.n_max <= pool, f"n_max={args.n_max} exceeds the train pool ({pool})"

    perm = np.random.default_rng(args.seed).permutation(pool)
    args.bin_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.bin_dir / f"perm_seed{args.seed}.npy", perm)

    manifest = {}
    for n in range(args.n_min, args.n_max + 1, args.n_step):
        out_dir = args.bin_dir / f"n{n}"
        if (out_dir / f"train.{SRC}-{TGT}.{TGT}.bin").exists():
            print(f"n={n}: exists, skipping")
        else:
            print(f"n={n}: binarising")
            binarise_subset(n, perm, src_lines, tgt_lines, args.full_bin, out_dir, args.workers)
        manifest[n] = str(out_dir)

    with open(args.bin_dir / "subsets.json", "w") as f:
        json.dump({"seed": args.seed, "pool": pool, "subsets": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
