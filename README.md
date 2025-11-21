# Federated Conformal Calibration (FCC) Code

Python reference implementation for the experiments described in the paper. It trains a global model with FedAvg, produces calibration baselines, and runs Federated Conformal Calibration with quantile fusion, blended client thresholds, and optional temperature scaling. Metrics include ECE, NLL, client-conditional coverage, dispersion, and prediction-set size.

## Requirements
Install in a fresh environment:
```
pip install -r requirements.txt
```

## Quick start (CIFAR-10)
Train a ResNet-50 with FedAvg on a Dirichlet $\alpha=0.3$ split, then run all calibrations and evaluations:
```
python run_experiment.py \
  --dataset cifar10 \
  --model resnet50 \
  --alpha 0.3 \
  --clients 10 \
  --rounds 20 \
  --local-epochs 1 \
  --batch-size 128 \
  --participation 0.5 \
  --output out/cifar10_resnet50_a03
```
Results (JSON + TSV) land in `--output`.

## Datasets
- CIFAR-10/100: downloaded automatically via `torchvision`.
- Tiny-ImageNet: point `--data-root` to a directory containing `train/` and `val/` in the standard layout.
- DomainNet: point `--data-root` to a pre-downloaded DomainNet root; provide `--domains` list (default: `clipart,painting,real,sketch`).

## Calibration modes
- Uncalibrated
- Global Temperature Scaling
- Per-client Temperature Scaling
- Per-client Split Conformal
- FCC (quantile fusion + blended thresholds + optional global temperature)

## Key options
- `--alpha`: Dirichlet concentration for label skew.
- `--blend-lambda`: FCC blend $\lambda$.
- `--quantiles`: comma list for FCC grid.
- `--trim`: trimming $\kappa$ for quantile fusion.
- `--beta`: cushion factor for small-sample correction.
- `--epsilon`: local DP noise scale (set 0 to disable).
- `--target`: target coverage (e.g., 0.9).

See `python run_experiment.py --help` for the full set of flags.

## Files
- `run_experiment.py`: end-to-end pipeline (train → calibrate → evaluate).
- `federated.py`: FedAvg training loop.
- `data.py`: dataset loading and non-iid partitioning.
- `models.py`: ResNet-50 and ViT-B/16 builders.
- `calibration.py`: FCC and baseline calibration utilities.
- `metrics.py`: ECE, NLL, coverage, dispersion, set-size, risk-coverage.
- `utils.py`: helpers for seeding, devices, state dict averaging, and logging.
