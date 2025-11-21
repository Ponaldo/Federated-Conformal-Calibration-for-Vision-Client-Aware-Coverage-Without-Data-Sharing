import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import to_numpy


def _forward_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_list.append(logits.detach().cpu())
            labels_list.append(y.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, init_temp: float = 1.0, max_iter: int = 50) -> float:
    """
    Standard temperature scaling on logits to minimize NLL.
    """
    temperature = torch.tensor([init_temp], requires_grad=True, device=logits.device)
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def _closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / temperature


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def _quantile_with_cushion(scores: np.ndarray, grid: Sequence[float], beta: float) -> Dict[float, float]:
    n = len(scores)
    quantiles = {}
    for tau in grid:
        adj = min(tau + beta / math.sqrt(max(n, 1)), 1.0)
        quantiles[tau] = float(np.quantile(scores, adj))
    return quantiles


def client_summary(
    model: nn.Module,
    loader: DataLoader,
    grid: Sequence[float],
    beta: float,
    device: torch.device,
    epsilon: float = 0.0,
) -> Tuple[int, Dict[float, float]]:
    """
    Compute nonconformity quantiles (1 - p_true) for a client.
    """
    logits, labels = _forward_logits(model, loader, device)
    probs = softmax_probs(logits)
    true_probs = probs[torch.arange(len(labels)), labels]
    scores = 1.0 - to_numpy(true_probs)
    n = len(scores)
    quantiles = _quantile_with_cushion(scores, grid, beta)
    if epsilon > 0:
        sensitivity = 1.0 / max(n, 1)
        for k in quantiles:
            noise = np.random.laplace(0.0, sensitivity / epsilon)
            quantiles[k] += float(noise)
    return n, quantiles


def weighted_trimmed_quantile(values: List[float], weights: List[float], tau: float, trim: float) -> float:
    """
    Compute weighted quantile with symmetric trimming by total weight fraction 'trim'.
    """
    arr = np.array(values)
    w = np.array(weights)
    order = arr.argsort()
    arr = arr[order]
    w = w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    lower = trim * total
    upper = (1 - trim) * total
    keep = (cum >= lower) & (cum <= upper)
    arr = arr[keep]
    w = w[keep]
    if len(arr) == 0:
        return float(np.mean(values))
    cum = np.cumsum(w)
    cum /= cum[-1]
    return float(np.interp(tau, cum, arr))


def enforce_monotone(grid: Sequence[float], values: List[float]) -> List[float]:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    fitted = iso.fit_transform(list(grid), values)
    return [float(v) for v in fitted]


def interpolate_on_grid(grid: Sequence[float], values: Sequence[float], target: float) -> float:
    return float(np.interp(target, grid, values))


@dataclass
class FCCConfig:
    grid: Tuple[float, ...]
    beta: float = 0.03
    rho: float = 0.75
    trim: float = 0.05
    blend_lambda: float = 0.6
    target: float = 0.9
    epsilon: float = 0.0  # local DP noise; 0 disables
    temperature: bool = True


def federated_quantile_fusion(
    counts: List[int],
    quantiles: List[Dict[float, float]],
    cfg: FCCConfig,
) -> Tuple[List[float], List[float]]:
    weights = [c ** cfg.rho for c in counts]
    fused = []
    grid = list(cfg.grid)
    for tau in grid:
        vals = [q[tau] for q in quantiles]
        fused.append(weighted_trimmed_quantile(vals, weights, tau, cfg.trim))
    fused = enforce_monotone(grid, fused)
    tau_star = 1.0 - cfg.target
    fused_tau_star = interpolate_on_grid(grid, fused, tau_star) if tau_star not in grid else fused[grid.index(tau_star)]

    thresholds = []
    for q in quantiles:
        local_tau_star = interpolate_on_grid(grid, [q[t] for t in grid], tau_star) if tau_star not in grid else q[tau_star]
        t_c = (1 - cfg.blend_lambda) * local_tau_star + cfg.blend_lambda * fused_tau_star
        thresholds.append(1.0 - t_c)
    return fused, thresholds


def compute_fcc_cards(
    model: nn.Module,
    calib_loaders: List[DataLoader],
    device: torch.device,
    cfg: FCCConfig,
) -> Tuple[List[float], List[float], List[float]]:
    counts: List[int] = []
    quantiles: List[Dict[float, float]] = []
    for loader in calib_loaders:
        n, q = client_summary(model, loader, cfg.grid, cfg.beta, device, epsilon=cfg.epsilon)
        counts.append(n)
        quantiles.append(q)
    fused, thresholds = federated_quantile_fusion(counts, quantiles, cfg)

    temp = 1.0
    if cfg.temperature:
        # Fit a single global temperature on pooled calibration logits
        logits_all = []
        labels_all = []
        for loader in calib_loaders:
            logits, labels = _forward_logits(model, loader, device)
            logits_all.append(logits)
            labels_all.append(labels)
        logits_cat = torch.cat(logits_all, dim=0).to(device)
        labels_cat = torch.cat(labels_all, dim=0).to(device)
        temp = fit_temperature(logits_cat, labels_cat)
    return fused, thresholds, [temp for _ in thresholds]


def per_client_conformal_thresholds(
    model: nn.Module,
    calib_loaders: List[DataLoader],
    device: torch.device,
    target: float,
) -> List[float]:
    tau_star = 1.0 - target
    thresholds = []
    for loader in calib_loaders:
        logits, labels = _forward_logits(model, loader, device)
        probs = softmax_probs(logits)
        true_probs = probs[torch.arange(len(labels)), labels]
        scores = 1.0 - to_numpy(true_probs)
        q = np.quantile(scores, tau_star)
        thresholds.append(1.0 - q)
    return thresholds


def global_temperature(
    model: nn.Module,
    calib_loaders: List[DataLoader],
    device: torch.device,
) -> float:
    logits_all = []
    labels_all = []
    for loader in calib_loaders:
        logits, labels = _forward_logits(model, loader, device)
        logits_all.append(logits)
        labels_all.append(labels)
    logits_cat = torch.cat(logits_all, dim=0).to(device)
    labels_cat = torch.cat(labels_all, dim=0).to(device)
    return fit_temperature(logits_cat, labels_cat)


def per_client_temperature(
    model: nn.Module,
    calib_loaders: List[DataLoader],
    device: torch.device,
) -> List[float]:
    temps = []
    for loader in calib_loaders:
        logits, labels = _forward_logits(model, loader, device)
        temps.append(fit_temperature(logits.to(device), labels.to(device)))
    return temps
