import copy
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import average_state_dicts, get_device


@dataclass
class FedAvgConfig:
    rounds: int = 100
    local_epochs: int = 1
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    participation: float = 0.5
    device: str = ""


def _local_train(
    model: nn.Module,
    loader: DataLoader,
    cfg: FedAvgConfig,
    device: torch.device,
) -> Tuple[dict, int]:
    model = copy.deepcopy(model)
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    n_samples = 0
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            n_samples += y.shape[0]
    return copy.deepcopy(model.state_dict()), n_samples


def fedavg_train(
    model: nn.Module,
    client_loaders: List[DataLoader],
    cfg: FedAvgConfig,
) -> nn.Module:
    device = get_device(cfg.device)
    num_clients = len(client_loaders)
    for rnd in range(cfg.rounds):
        k = max(1, int(num_clients * cfg.participation))
        active = random.sample(range(num_clients), k=k)
        states = []
        weights = []
        for idx in active:
            state, n = _local_train(model, client_loaders[idx], cfg, device)
            states.append(state)
            weights.append(float(n))
        new_state = average_state_dicts(states, weights)
        model.load_state_dict(new_state)
    model.to(device)
    return model
