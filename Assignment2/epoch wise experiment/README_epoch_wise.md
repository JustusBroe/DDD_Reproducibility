# Epoch-wise double descent

Reproduces Sec. 6 of Nakkiran et al., *Deep Double Descent* (ICLR 2020): Fig. 9 and Fig. 10.

| Paper figure | Setting | Our runs (`ucloud/plan.txt`) |
|---|---|---|
| Fig. 9 left | ResNet18, CIFAR-10, 20 % noise, small / intermediate / large model (paper: k = 3, 12, 64) | k = 4, 12, 32, 64 |
| Fig. 9 right (heatmap, all widths x epochs) | 64 widths x 4K epochs | drawn from the authors' raw data, our widths marked |
| Fig. 10a | ResNet18 k=64, CIFAR-10, 0/10/20 % noise | 3 runs (20 % shared with Fig. 9) |
| Fig. 10b | ResNet18 k=64, CIFAR-100, 0/10/20 % noise | 3 runs |
| Fig. 10c | 5-layer CNN k=128, CIFAR-10, SGD, 0/10/20 % noise | 3 runs |

Fig. 2 of the paper shows the same kind of heatmap at **15 %** noise; it is not part of this reproduction.

## Files

| File | Purpose |
|---|---|
| `resnet18k.py`, `CNN.py` | Model definitions, verbatim from the authors' repo |
| `data.py` | CIFAR-10/100 + fixed label noise (sampled once from `--noise-seed`), GPU-side RandomCrop(32, pad 4) + flip |
| `train.py` | Trains one run; logs every epoch to `runs/<name>/log.csv`; resumes from `last.pt` |
| `fetch_reference.py` | Downloads the authors' raw per-epoch curves from `gs://hml-public/dd/` into `reference/` |
| `plot_epochwise.py` | Our curves (solid) vs. paper curves (dashed), modes `widths`, `fig10`, `all` |
| `summarize.py` | Table of first minimum / peak / final error / interpolation epoch, ours vs. paper -> `results/summary.{csv,tex}` |
| `ucloud/setup.sh`, `ucloud/run_epochwise.sh`, `ucloud/plan.txt` | UCloud setup, queue runner, list of runs |
| `ucloud/run_full.sh` | UCloud batch entry point used for the reported results (`PER_GPU=12`, `ADAM_EPOCHS=4000`) |

## Reproducing everything

```bash
bash ucloud/setup.sh            # environment, CIFAR-10/100, paper curves
bash ucloud/run_full.sh         # 12 runs, then figures/
python summarize.py             # results/summary.csv + summary.tex
```

The reported runs took 22 h wall-clock, all 12 in parallel on one NVIDIA B200 (UCloud, 2026-10-07/08).
Committed outputs: `runs/<name>/{log.csv,config.json,flipped_idx.npy}`, `logs/`, `figures/`, `results/`.
Checkpoints (`*.pt`), datasets and the raw paper curves are not committed (see `.gitignore`).

## Setup (paper Sec. 4 + App. B)

* **ResNet18** (widths [k, 2k, 4k, 8k], k=64 standard, 11.2M params): Adam, constant lr 1e-4, 4000 epochs.
* **5-layer CNN** (k=128, 6.2M params; k=64 has 1,558,026 params, identical to the paper): SGD without
  momentum, lr(t) = 0.1 / sqrt(1 + floor(t/512)) at gradient step t, 500K steps (1279 epochs).
* Both: batch 128, cross-entropy, default PyTorch init, no weight decay / dropout, RandomCrop(32, pad 4) +
  horizontal flip. Label noise p: each train label is replaced w.p. p by a uniformly random **incorrect**
  label, sampled once. Test error is on the **clean** test set, as in the paper's figures (except Fig. 1).

`log.csv` columns: `step`, `lr`, `train_loss/train_err` (running, augmented batches, noisy labels),
`test_loss/test_err` (clean test set, every epoch), and every 10 epochs `full_train_err` (eval mode, no aug,
noisy labels), `noisy_fit` (fraction of mislabeled samples predicted with their noisy label) and
`clean_train_err`. Epochs derived from these (interpolation, memorisation) therefore have 10-epoch resolution.

## Reference data (authors' raw curves)

| Folder | Used for | Notes |
|---|---|---|
| `cifar10-resnet18k-50k-adam` | Fig. 9, 10a | entries 0/1/2 = 0/10/20 % noise; k = 1..64; 3999 epochs |
| `cifar100-resnet18k-50k-adam` | Fig. 10b | same layout |

The files are not labelled; the noise level of each entry was inferred from its final error and checked
against the published Fig. 10a/b (final errors and interpolation epochs agree). **Fig. 10c has no public
raw data**: the README of the authors' repo links its epoch-wise data to a private bucket, and the public
CNN/SGD files (`cifar10-mcnn-p10-sgd`, `pct-cifar10-mcnn-50000-p{0,20}-sgd-big`, used for the sample-wise
experiments) do not match the published panel (e.g. 37 % vs. ~22 % final error at 20 % noise). The CNN runs
are therefore compared with the published figure only.

## Deviations from the paper (state in the report)

* Fig. 9 "small model": k = 4 instead of k = 3 (both are in the monotone regime in the paper's own data).
* One seed per setting (the paper's epoch-wise figures are also single runs).
* Augmentation is implemented as batched GPU ops; it is the same RandomCrop(32, pad 4) + flip distribution.
* Inputs are normalised with CIFAR-10 mean/std (also for CIFAR-100); the paper does not state its normalisation.
* Label noise is sampled once with a fixed seed (Bernoulli per sample; realised 10.19 % / 20.03 % for
  p = 0.1 / 0.2). The 10 % noisy set is a subset of the 20 % one, and identical across datasets/models.
