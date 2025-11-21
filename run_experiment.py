import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from data import (
    DATASETS,
    build_client_loaders,
    build_test_loader,
    default_transforms,
    dirichlet_partition,
    labels_from_dataset,
    load_base_dataset,
    split_client_data,
)
from federated import FedAvgConfig, fedavg_train
from models import build_model
from calibration import (
    FCCConfig,
    compute_fcc_cards,
    global_temperature,
    per_client_conformal_thresholds,
    per_client_temperature,
)
from metrics import (
    aggregate_client_metrics,
    collect_logits,
    compute_ece,
    compute_nll,
    evaluate_clients_sets,
    evaluate_global,
)
from utils import get_device, save_json, set_seed
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Conformal Calibration experiments")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=DATASETS.keys())
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--domains", type=str, default="clipart,painting,real,sketch")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit-b16"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--participation", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration for label skew")
    parser.add_argument("--calib-frac", type=float, default=0.1)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output", type=str, default="out/run")
    parser.add_argument("--quantiles", type=str, default="0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.94,0.95,0.97,0.99")
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--trim", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.75)
    parser.add_argument("--blend-lambda", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=0.0, help="LDP noise scale; 0 disables noise")
    parser.add_argument("--no-temp", action="store_true", help="Disable temperature fitting in FCC")
    return parser.parse_args()


def build_eval_loaders(subsets: Sequence[Subset], batch_size: int) -> List[DataLoader]:
    return [
        DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False) for sub in subsets
    ]


def pooled_probs_and_labels(
    model,
    loaders: Sequence[DataLoader],
    device: torch.device,
    temperatures: Sequence[float],
) -> torch.Tensor:
    logits_all = []
    labels_all = []
    for loader, temp in zip(loaders, temperatures):
        logits, labels = collect_logits(model, loader, device, temperature=temp)
        logits_all.append(logits)
        labels_all.append(labels)
    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    probs = torch.softmax(logits_cat, dim=-1)
    return probs, labels_cat


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = DATASETS[args.dataset]
    domains = args.domains.split(",")
    grid = tuple(float(x) for x in args.quantiles.split(","))

    train_tf = default_transforms(spec, train=True)
    test_tf = default_transforms(spec, train=False)

    train_ds = load_base_dataset(args.dataset, args.data_root, train=True, transform=train_tf, domains=domains)
    test_ds = load_base_dataset(args.dataset, args.data_root, train=False, transform=test_tf, domains=domains)

    # Partition training data
    train_labels = labels_from_dataset(train_ds)
    client_indices = dirichlet_partition(train_labels, args.clients, args.alpha, seed=args.seed)
    train_subsets, calib_subsets = split_client_data(
        train_ds, client_indices, calib_fraction=args.calib_frac, seed=args.seed
    )
    train_loaders, calib_loaders = build_client_loaders(
        train_subsets, calib_subsets, batch_size=args.batch_size, num_workers=2
    )

    # Partition test data for per-client evaluation
    test_labels = labels_from_dataset(test_ds)
    test_indices = dirichlet_partition(test_labels, args.clients, args.alpha, seed=args.seed + 1)
    test_subsets = [Subset(test_ds, idxs) for idxs in test_indices]
    test_loaders = build_eval_loaders(test_subsets, batch_size=args.batch_size)
    global_test_loader = build_test_loader(test_ds, batch_size=args.batch_size, num_workers=2)

    # Build and train model
    model = build_model(args.model, spec.num_classes, pretrained=args.pretrained)
    fed_cfg = FedAvgConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        participation=args.participation,
        device=args.device,
    )
    model = fedavg_train(model, train_loaders, fed_cfg)
    model.to(device)

    results: Dict = {"config": vars(args)}

    # Uncalibrated global metrics
    results["global_uncal"] = evaluate_global(model, global_test_loader, device)

    # Global temperature scaling
    temp_global = global_temperature(model, calib_loaders, device)
    results["global_temp"] = {"temp": temp_global, **evaluate_global(model, global_test_loader, device, temp_global)}

    # Per-client temperature scaling
    temps_pc = per_client_temperature(model, calib_loaders, device)
    logits_cat_list = []
    labels_cat_list = []
    for loader, temp in zip(test_loaders, temps_pc):
        logits, labels = collect_logits(model, loader, device, temperature=temp)
        logits_cat_list.append(logits)
        labels_cat_list.append(labels)
    logits_pc = torch.cat(logits_cat_list, dim=0)
    labels_pc = torch.cat(labels_cat_list, dim=0)
    probs_pc = torch.softmax(logits_pc, dim=-1)
    results["per_client_temp"] = {
        "temps": temps_pc,
        "ece": compute_ece(probs_pc, labels_pc),
        "nll": compute_nll(logits_pc, labels_pc),
        "acc": float((probs_pc.argmax(dim=1) == labels_pc).float().mean().item()),
    }

    # Per-client conformal sets
    thresholds_pcp = per_client_conformal_thresholds(model, calib_loaders, device, target=args.target)
    pcp_metrics = evaluate_clients_sets(
        model, test_loaders, device, thresholds_pcp, temperatures=[1.0] * len(thresholds_pcp)
    )
    results["pcp"] = {
        "thresholds": thresholds_pcp,
        **aggregate_client_metrics(pcp_metrics, target=args.target),
    }

    # FCC
    fcc_cfg = FCCConfig(
        grid=grid,
        beta=args.beta,
        rho=args.rho,
        trim=args.trim,
        blend_lambda=args.blend_lambda,
        target=args.target,
        epsilon=args.epsilon,
        temperature=not args.no_temp,
    )
    fused_quantiles, thresholds_fcc, temps_fcc = compute_fcc_cards(model, calib_loaders, device, fcc_cfg)
    fcc_metrics = evaluate_clients_sets(model, test_loaders, device, thresholds_fcc, temps_fcc)
    results["fcc"] = {
        "fused_quantiles": fused_quantiles,
        "thresholds": thresholds_fcc,
        "temps": temps_fcc,
        **aggregate_client_metrics(fcc_metrics, target=args.target),
    }

    os.makedirs(args.output, exist_ok=True)
    save_json(results, os.path.join(args.output, "results.json"))

    # Save per-client set metrics for PCP and FCC
    def _to_rows(name: str, data: List[Dict[str, float]]):
        return [{"method": name, "client": i, **m} for i, m in enumerate(data)]

    rows = _to_rows("pcp", pcp_metrics) + _to_rows("fcc", fcc_metrics)
    pd.DataFrame(rows).to_csv(os.path.join(args.output, "client_set_metrics.tsv"), sep="\t", index=False)
    print("Finished. Results saved to", args.output)


if __name__ == "__main__":
    main()
