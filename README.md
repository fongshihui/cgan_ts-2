# Conditional LSTM GAN for Financial Residuals

This project implements a **Conditional LSTM GAN** to generate standardized
financial residual sequences conditioned on volatility regimes.

## End-to-End Flow

1. Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

2. Prepare training data from CSV (must contain `Close`)
```bash
python3 data_prep.py --csv /path/to/your_prices.csv --out data.npz --seq_len 100
```

What this does:
- Reads close prices and computes log returns.
- Fits `GARCH(1,1)` on returns.
- Computes standardized residuals `eps = r / sigma`.
- Builds rolling windows of length `seq_len`.
- Buckets each window into 3 volatility regimes (`0/1/2`) using quantiles.
- Saves `eps` and `cond` into `data.npz`.

3. Train conditional GAN
```bash
python3 train.py
```

What training does:
- `Generator(z, cond)` produces fake residual windows.
- `Discriminator(x, cond)` predicts real vs fake.
- Adversarial training updates both models over dataset windows.
- Saves generator weights to `generator.pt` at the end.

4. Generate and visualize samples
```bash
python3 eval.py
```

What eval does:
- Loads trained `Generator`.
- Samples noise + regime labels.
- Applies optional generation controls.
- Plots synthetic residual sequences.

Generation controls:
- `--desired_volatility` in `[0.5, 2.0]` (scales amplitude)
- `--desired_trend` in `[-1.0, 1.0]` (down/up directional drift)
- `--desired_fat_tails` in `[0.5, 2.0]` (tail amplification/compression)
- `--desired_momentum` in `[0.0, 1.0]` (persistence)

Example:
```bash
python3 eval.py \
  --n_samples 6 \
  --seq_len 100 \
  --conds 0,1,2,0,1,2 \
  --desired_volatility 1.4 \
  --desired_trend 0.3 \
  --desired_fat_tails 1.5 \
  --desired_momentum 0.7
```

Reconstruct returns and prices (using GARCH sigma forecast from a CSV):
```bash
python3 eval.py \
  --csv /path/to/your_prices.csv \
  --n_samples 6 \
  --seq_len 100 \
  --conds 0,1,2,0,1,2 \
  --desired_volatility 1.4 \
  --desired_trend 0.3 \
  --desired_fat_tails 1.5 \
  --desired_momentum 0.7 \
  --out_npz synthetic_paths.npz \
  --out_ohlcv_dir synthetic_ohlcv \
  --run_quality_eval \
  --quality_out_json quality_report.json
```

`synthetic_paths.npz` contains:
- `eps`: generated standardized residuals
- `returns`: reconstructed log returns (`eps * sigma`)
- `prices`: reconstructed synthetic price paths
- `sigma`: forecast volatility path used for reconstruction
- `cond`: class labels used for each sample

`synthetic_ohlcv/` contains one CSV per generated sample:
- columns: `step, Open, High, Low, Close, Volume`

When `--run_quality_eval` is enabled, eval also writes:
- `quality_report.json`: distribution, VaR/ES, ACF, regime stats, and NN distance metrics

## Quality Evaluation

Evaluate synthetic quality against a real CSV:

```bash
python3 evaluate_quality.py \
  --csv sample_prices.csv \
  --npz synthetic_paths.npz \
  --out_json quality_report.json
```

It reports:
- distribution stats (mean/std/skew/kurtosis/quantiles)
- VaR/ES at 95% and 99%
- ACF of returns and absolute returns
- regime-wise residual behavior by class
- nearest-neighbor L2 distance (memorization signal)

## Quickstart With Included Dataset

A sample dataset is included at `sample_prices.csv` (column: `Close`).

```bash
python3 data_prep.py --csv sample_prices.csv --out data.npz --seq_len 100
python3 train.py
python3 eval.py --csv sample_prices.csv --out_npz synthetic_paths.npz
```

## Training vs GitHub

- Training quality improves only when you run `train.py` (or fine-tune) on data.
- Pushing to GitHub does not train the model and does not improve weights automatically.
- GitHub helps with:
- versioning code and experiment setup
- sharing reproducible pipeline and configs
- collaboration via pull requests/issues

How others benefit:
- They can clone the repo and run the same training/eval flow.
- They can train on their own data for new checkpoints.
- If you share `generator.pt`, they can use that exact trained model directly.

## Pipeline Summary
1. Close prices -> log returns
2. Fit GARCH(1,1)
3. Extract standardized residuals
4. Train conditional GAN on residual windows
5. Sample synthetic residuals by volatility regime
