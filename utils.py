import json
import os
import random
from typing import Dict, Iterable, List, Tuple

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


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
