# CIFAR-10 quick evaluation (pretrained ResNet-50)

Command:
```
python run_experiment.py --dataset cifar10 --model resnet50 --pretrained --clients 5 --rounds 0 --local-epochs 1 --batch-size 128 --participation 1 --alpha 0.3 --calib-frac 0.1 --target 0.9 --output out/cifar10_pretrained --train-fraction 0.05 --test-fraction 0.1 --quantiles 0.5,0.7,0.9
```

Classification metrics (global logits):

| Method | Acc | ECE | NLL | Notes |
| --- | --- | --- | --- | --- |
| Uncalibrated | 0.0880 | 0.1711 | 2.6016 | Direct evaluation of pretrained weights without federated updates |
| Global Temp | 0.0880 | 0.0391 | 2.3221 | Learned scalar temperature 4.70 over all calibration data |
| Per-client Temp | 0.0880 | 0.0395 | 2.3136 | Client-specific temperatures [4.29, 6.30, 7.23, 4.79, 4.68] |

Conformal set metrics (target coverage = 0.9):

| Method | Mean Coverage | Coverage Abs Error | Coverage Std | Mean Set Size | Notes |
| --- | --- | --- | --- | --- | --- |
| Per-client Split Conformal | 0.1234 | 0.7766 | 0.0626 | 1.4516 | Thresholds per client [0.152, 0.172, 0.186, 0.185, 0.155] |
| Federated Conformal Calibration | 0.7692 | 0.1435 | 0.1649 | 7.8587 | Fused quantiles [0.908, 0.939, 0.970]; thresholds per client [0.0842, 0.0926, 0.0824, 0.0912, 0.0886]; shared temperature 4.98 |

Per-client conformal metrics are in `out/cifar10_pretrained/client_set_metrics.tsv` (ignored by git). Overall aggregates are recorded in `out/cifar10_pretrained/results.json`.
