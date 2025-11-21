from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .calibration import apply_temperature, softmax_probs


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    labels = labels.to(probs.device)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(num_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            acc = accuracies[in_bin].float().mean()
            conf = confidences[in_bin].mean()
            ece += torch.abs(acc - conf) * prop
    return float(ece.item())


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = nn.CrossEntropyLoss()(logits, labels)
    return float(loss.item())


def collect_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device, temperature: float = 1.0):
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if temperature != 1.0:
                logits = apply_temperature(logits, temperature)
            logits_list.append(logits.detach().cpu())
            labels_list.append(y.detach().cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def prediction_set_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> Dict[str, float]:
    """
    Given probabilities and labels, compute coverage, mean set size, and selectivity.
    """
    indicators = (probs >= threshold).float()
    set_sizes = indicators.sum(dim=1)
    covered = indicators[torch.arange(len(labels)), labels] > 0
    coverage = covered.float().mean().item()
    mean_size = set_sizes.float().mean().item()
    return {"coverage": coverage, "set_size": mean_size}


def evaluate_global(model, loader: DataLoader, device: torch.device, temperature: float = 1.0) -> Dict[str, float]:
    logits, labels = collect_logits(model, loader, device, temperature)
    probs = torch.softmax(logits, dim=-1)
    return {
        "ece": compute_ece(probs, labels),
        "nll": compute_nll(logits, labels),
        "acc": float((probs.argmax(dim=1) == labels).float().mean().item()),
    }


def evaluate_clients_sets(
    model,
    loaders: Sequence[DataLoader],
    device: torch.device,
    thresholds: Sequence[float],
    temperatures: Sequence[float],
) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for loader, thr, temp in zip(loaders, thresholds, temperatures):
        logits, labels = collect_logits(model, loader, device, temperature=temp)
        probs = torch.softmax(logits, dim=-1)
        m = prediction_set_metrics(probs, labels, thr)
        metrics.append(m)
    return metrics


def aggregate_client_metrics(client_results: Sequence[Dict[str, float]], target: float) -> Dict[str, float]:
    coverages = np.array([m["coverage"] for m in client_results])
    sizes = np.array([m["set_size"] for m in client_results])
    return {
        "coverage_mean": float(coverages.mean()),
        "coverage_abs_error": float(np.abs(coverages - target).mean()),
        "coverage_std": float(coverages.std()),
        "set_size": float(sizes.mean()),
    }
