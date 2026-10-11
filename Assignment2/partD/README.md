# Part D: label-noise ablation (Pijus)

Report section *Label-noise ablation*. These experiments reproduce four figures of Nakkiran et al., *Deep Double Descent* (arXiv:1912.02292), and add one extension. All of them ask the same question: does the double-descent peak come from fitting noisy labels, as the paper claims?

| ID | Paper figure | Setting | Runs |
|---|---|---|---|
| E5 | Fig. 4b | ResNet18, CIFAR-10, augmentation, Adam (lr 1e-4), 500 epochs. Label noise p ∈ {0, 5, 10%}; 15% comes from E3 member 0 and 20% from E1b (λ = 0) | 66 |
| E1 | Fig. 21 | 5-layer CNN, 10% noise, augmentation, SGD (inverse-sqrt lr, γ0 = 0.1), 500K steps (1280 epochs). Weight decay λ ∈ {0, 1e-4, 5e-4}. Two extra seeds at k ∈ {16, 24, 32, 40} for λ ∈ {0, 5e-4}, used only for error bars | 82 |
| E2 | Fig. 22 | ResNet18 at k = 64, 20% noise, augmentation, SGD (inverse-sqrt lr), 500 epochs. 14 values of λ from 0 to 0.1 | 14 |
| E3 | Fig. 28 | ResNet18, 15% noise, Adam, 500 epochs. 5 members per width, each with its own noise draw and initialisation, combined by plurality vote | 75 |
| E1b | extension | ResNet18, 20% noise, Adam, 500 epochs, λ ∈ {0, 5e-4} | 44 |
| E1n | Fig. 21 check | E1 with λ = 0, trained 1459 epochs like the authors' released λ = 0 run, test set also scored on 10%-noisy test labels every epoch | 22 |

As in the paper, there is one run per setting (seed 0, noise seed 0). The only exceptions are the ensemble members and the extra seeds in E1.

## Requirements
Python 3 with PyTorch, NumPy, pandas and matplotlib (`pip install torch numpy pandas matplotlib`). Training needs a CUDA GPU; we trained with PyTorch 2.12 on NVIDIA B200 GPUs and ran the analysis with PyTorch 2.14, NumPy 2.5, pandas 3.0 and matplotlib 3.11. CIFAR-10 is downloaded to `data/` on first use.

## Files
- `resnet18k.py`, `cnn5.py`: the authors' model definitions, unchanged (the same as `resnet18k.py` and `CNN.py` at the repository root).
- `common.py`: CIFAR-10 kept on the GPU (uint8), batched GPU augmentation, the label-noise function and the evaluation code.
  - Noise: with probability p a label becomes `(y + U{1..9}) mod 10`, i.e. a uniformly random wrong class. The draw is fixed by the noise seed.
- `train.py`: trains one model (`--test-noise p` also scores the test set on labels with noise p, logged as `test_*_noisy`). It writes `config.json`, `metrics.csv` (one row per epoch, plus a full train-set evaluation every 10 epochs), `summary.json` and `test_logits.npz`. A `ckpt.pt` written every 25 epochs lets an interrupted run resume; it is deleted when the run finishes. `--max-minutes` stops cleanly at a checkpoint, for jobs with a time limit.
- `run_grid.py`: defines all 303 runs (E1, E2, E3, E1b, E5, E1n) and runs them, several at a time on every visible GPU (or the CPU).
  - `--capacity` is the load per GPU; a run costs 1–4 units depending on the model size.
  - Finished runs (with `summary.json`) are skipped and interrupted runs resume, so the same command can simply be started again.
  - `--only-k` selects widths, e.g. to split the grid over several machines or jobs.
- `authors_data.py`: downloads the authors' raw results from `gs://hml-public/dd` and decodes them with a restricted unpickler that only allows numpy objects.
- `colormap_inferno_strong_1.txt`: the authors' heatmap colormap (from `double-descent/custom_colormaps/`), used for Fig. 22.
- `analyze.py`: computes the metrics and writes `figures/` and `tables/`.
- `check_e1n.py`: compares E1n with the authors' released λ = 0 run at epochs 1280 and 1459 (`figures/fig21_noisy_check`, `tables/e1n_check.csv`).
- `figures/`: `fig4b_label_noise`, `fig21_weight_decay_cnn`, `fig22_weight_decay_sweep`, `fig28_ensembling`, `figExt_resnet_adam_weight_decay` and `fig21_noisy_check`, each as .pdf and .png.
- `tables/`: `metrics.csv`/`.tex` (all experiments), `noise_metrics.csv` (Fig. 4b, ours and the authors'), `ensemble.csv`, `e1n_check.csv` and `runs_E*.csv` (final metrics of every run).
- `results/<exp>/<run>/`: `config.json`, `metrics.csv` and `summary.json` for each of our runs. `test_logits.npz` is kept only for E3, where the ensemble needs it.

## Reproduce
The figures and tables can be rebuilt from our results in `results/` without training:
```bash
python authors_data.py
python analyze.py
python check_e1n.py
```
`authors_data.py` downloads the authors' raw results to `authors_data/` (the dashed curves). `analyze.py` writes `figures/` and `tables/`, and `check_e1n.py` adds the E1n comparison.

To train everything again, move our `results/` folder away first (finished runs are skipped), then:
```bash
python run_grid.py --dry-run
python run_grid.py --capacity 8
python run_grid.py --exps E1 E1n --capacity 8
```
- Without `--exps` all six experiments run. `--dry-run` lists the runs that are not finished yet. Each run logs to `results/logs/`.
- On an HPC cluster, submit `python run_grid.py ...` as a batch job and submit it again if the job hits its time limit. To use several jobs at once, give each job different widths with `--only-k`.
- One run on its own, e.g. a single point of Fig. 21:
  ```bash
  python train.py --arch cnn5 --k 24 --noise 0.1 --wd 5e-4 --opt sgd --lr 0.1 --lr-sched invsqrt --epochs 1280 --out results/E1/cnn5_k24_p0.10_wd5e-04_sgd_s0_n0
  ```
- Training time, summed over runs while several runs shared one B200: E1 69 h, E1n 25 h, E2 7 h, E3 37 h, E5 35 h, E1b 25 h.

## Metrics
- Interpolation threshold `k_0.1`: the smallest width whose train error on the noisy labels is at most 10%.
- Peak height `h`: the largest rise of the test error above its running minimum. It is searched only where the train error has fallen below half its value at k = 1. A peak is counted when h ≥ 1.5 percentage points. `k*` is the width where the peak occurs.
- Ensemble gain Δ(k): the mean single-model error minus the ensemble error.

## Deviations from the paper
- **ResNet training length:** 500 epochs instead of 4000 (TA-approved). The CNN gets the full 500K SGD steps.
- **Numerics:** bf16 autocast and channels_last. The paper does not report its precision.
- **Width grid:** k ≤ 64, with gaps above k = 12. The paper goes to 128 for the CNN.
- **Fig. 22:** the paper gives the data, model and optimiser (ResNet18, 20% noise, SGD with inverse-sqrt lr) but not the width or the training length. We use k = 64 and 500 epochs.
- **Weight decay:** coupled L2 (`torch.optim.*(weight_decay=λ)`), applied to all parameters.

## Results
Errors are in %. Differences between errors are in percentage points (pp).

### Fig. 4b: noise level and the clean baseline
Test error at the end of training. Ours: 500 epochs. Authors' released runs: 4000 epochs (their 15% value is the mean of five runs).

| p | ours: k_0.1 / h / k* / error at k=64 | authors: k_0.1 / h / k* / error at k=64 |
|---|---|---|
| 0% | 6 / 0.4 (no peak) / – / 6.9 | 5 / 0.7 (no peak) / – / 6.6 |
| 5% | 9 / 1.0 (plateau) / – / 11.6 | 7 / 2.3 / 9 / 10.7 |
| 10% | 11 / 3.1 / 12 / 14.2 | 9 / 7.4 / 9 / 12.9 |
| 15% | 14 / 5.6 / 12 / 17.7 | 10 / 8.4 / 10 / 16.3 |
| 20% | 14 / 7.6 / 14 / 21.0 | 11 / 11.2 / 10 / 19.6 |

- **Clean baseline:** the clean curve falls from 24.9% at k = 3 to 6.9% at k = 64 with no peak. The noisy curves approach it from above.
- **Peak height:** it grows with the noise level for both us and the authors.
- **Peak location:** the peak stays within two widths of k_0.1.
- **Difference from the authors:** our peaks are lower and further right. This is expected from 8 times shorter training.

### Fig. 21: weight decay, CNN
- **Train error:** the curves are identical for every λ (k_0.1 = 24).
- **Test-error peak:** it stays at k = 24–28: 25.9 / 26.2 / 24.5% for λ = 0 / 1e-4 / 5e-4.
- **Large models:** error at k = 64 is 18.8 / 17.4 / 16.0%. The test-loss peak is 1.12 / 1.02 / 0.85.
- **Optimal early stopping:** no peak for any λ.
- **Comparison:** our test error matches the published figure to about 1 pp. For λ > 0 it also matches the released runs.
- **The released λ = 0 run** (`pct-cifar10-mcnn-mdd-50k-p10-sgd-aug-fast`, 1459 epochs) is scored on noisy test labels.
  - Its test error is about 7 pp higher than ours and its test loss peaks at 2.75; the published loss curve peaks at 2.5.
  - E1n retrains our λ = 0 models for 1459 epochs and scores them every epoch on a fixed draw of 10%-noisy test labels.
    Over the 22 widths its test error matches the released run to 0.0 ± 1.2 pp at epoch 1280 (500K steps) and 0.2 ± 1.1 pp at epoch 1459.
  - Test loss: within 0.01 on average for k < 28 (each width within 0.06). For k ≥ 28 ours is 0.07 lower at both epochs (0.11 at k = 32 and 40).
    For comparison, the sd over noisy-label draws is 0.02–0.05 and over training seeds 0.01–0.06, so the residual is small but has the same sign at four of five widths.
  - On clean labels the E1n runs reproduce the original E1 λ = 0 runs at epoch 1280 to −0.3 ± 0.6 pp.
  - Mapping our clean error e to the noisy test distribution, e' = p + e(1 − 10p/9), gives the same answer as E1n for the error (0.2 ± 1.1 pp).
  - The released Fig. 11a data for n = 50k match the same mapped curve to −0.1 ± 0.6 pp.
- **In `fig21_weight_decay_cnn`** the λ = 0 curves are the E1n runs on noisy test labels, like the authors' released λ = 0 run; λ > 0 is on clean labels, like theirs. `tables/metrics.csv` uses clean labels for every λ.

### Fig. 22: weight-decay sweep at k = 64
- **λ = 0.1:** test error 84%.
- **First minimum:** 22.1% at 1/λ = 20. Only 3.5% of the wrong labels are fitted.
- **Peak:** 27.0% at 1/λ = 200. 91% of the wrong labels are fitted and the train error is 5.5%.
- **Second minimum:** 17.7% at 1/λ = 1000, where the training set is fitted.
- **λ = 0:** 19.7%.
- **Epoch-wise double descent** appears for every λ ≤ 1e-3.
- **Plotting note:** the paper's right panel is labelled "Train Loss" but shows train error, so we plot train error.

### Fig. 28: ensembling, 15% noise
- **Single model** (mean of 5): minimum 21.7% at k = 6, peak 26.9 ± 0.6% at k = 14 (= k_0.1), 17.4% at k = 64.
- **Ensemble:** no peak (largest rise 0.2 pp), 8.7% at k = 64.
- **Gain:** largest at the threshold, 13.3 pp at k = 14, compared with 6.4 pp at k = 6 and 8.7 pp at k = 64.

### Extension: Adam with coupled L2, 20% noise
- **Peak:** 30.4% at k = 14 without decay and 30.5% at k = 18 with λ = 5e-4.
- **Large models:** 22.3 vs 21.0% at k = 64. With decay the model no longer fits the training set (train error 2.0%).
- **Conclusion:** coupled L2 under Adam moves the peak instead of removing it.
