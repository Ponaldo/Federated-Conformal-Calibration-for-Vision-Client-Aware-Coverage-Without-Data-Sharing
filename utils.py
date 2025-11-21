import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "") -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_state_dicts(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No state dicts provided.")
    total = {}
    norm = sum(weights)
    for key in states[0].keys():
        total[key] = sum(w * s[key].to(torch.float32) for s, w in zip(states, weights)) / norm
    return total


def _to_serializable(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [
            _to_serializable(v) for v in obj
        ]  # preserve tuple contents; outer type not critical for JSON
    if isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (torch.device,)):
        return str(obj)
    return obj


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(obj), f, indent=2)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
