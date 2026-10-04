"""Merge results/*.json into results/results.csv (same leading columns as the authors' CSVs)."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent))
COLUMNS = ["n", "size", "test_error", "test_loss", "train_error", "train_loss"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--min-updates", type=int, default=80000,
                   help="ignore shorter runs, e.g. smoke tests")
    args = p.parse_args()

    rows = []
    for path in sorted(args.result_dir.glob("d*_n*_s*.json")):
        with open(path) as f:
            row = json.load(f)
        if row["num_updates"] >= args.min_updates:
            rows.append(row)
    if not rows:
        raise SystemExit(f"no finished runs in {args.result_dir}")

    df = pd.DataFrame(rows).sort_values(["size", "n", "seed"])
    extra = [c for c in df.columns if c not in COLUMNS]
    df = df[COLUMNS + extra]
    out = args.result_dir / "results.csv"
    df.to_csv(out, index=False)
    print(df[COLUMNS + ["seed", "num_params"]].to_string(index=False))
    print(f"\n{len(df)} runs -> {out}")


if __name__ == "__main__":
    main()
