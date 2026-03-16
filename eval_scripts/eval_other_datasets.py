#!/usr/bin/env python3
"""Evaluate HDC (quantized-aware and standard OnlineHD) on a single dataset.

Usage:
    python eval_other_datasets.py <ISOLET|UCIHAR> [onlineHD]

Produces three 'Testing accuracy of XX.XXX%' lines, matching the output
format that run_benchmark_other_datasets.py parses.
"""

import sys
import json
import pathlib
import math

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from tqdm import tqdm

import torchhd
import torchhd.datasets as datasets
from torchhd import embeddings

script_dir = pathlib.Path(__file__).parent.resolve()
repo_root = script_dir.parent
project_dir = repo_root / "mnist_example"

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from quant_models import Centroid

with open(project_dir / "config.json") as f:
    d = json.load(f)

DIMENSIONS = d["DIMENSIONS"]
NUM_LEVELS = d["NUM_LEVELS"]
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Encoder ─────────────────────────────────────────────────────────────────

class FlatLevelEncoder(nn.Module):
    """Base-level encoder for flat, real-valued feature vectors."""

    def __init__(self, in_features: int, out_features: int, levels: int):
        super().__init__()
        self.position = torchhd.embeddings.Random(in_features, out_features)
        self.value = torchhd.embeddings.Level(levels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features), float in [0, 1]
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


# ─── Dataset loading ─────────────────────────────────────────────────────────

def get_loaders(dataset_cls, data_root: str):
    train_ds = dataset_cls(data_root, train=True,  download=True)
    test_ds  = dataset_cls(data_root, train=False, download=True)

    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_ld = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    sample_x, _ = train_ds[0]
    num_features = int(sample_x.numel())
    num_classes  = int(train_ds.targets.max().item()) + 1

    return train_ld, test_ld, num_features, num_classes


# ─── Train / test ─────────────────────────────────────────────────────────────

def batch_encode(encode, loader):
    """Pre-encode entire dataset; returns (all_hv, all_labels) on device."""
    all_hv, all_labels = [], []
    with torch.no_grad():
        for samples, labels in tqdm(loader, desc="Encoding"):
            samples = samples.float().to(device)
            all_hv.append(encode(samples))
            all_labels.append(labels.to(device))
    return torch.cat(all_hv), torch.cat(all_labels)


def train(encode, model, train_ld):
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.float().to(device)
            labels  = labels.to(device)
            model.add(encode(samples), labels)


def test(encode, model, test_ld, num_classes):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    model.normalize(quantize=True)          # binarises weights in-place
    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc="Testing"):
            samples = samples.float().to(device)
            outputs = model(encode(samples), dot=True)
            accuracy.update(outputs.cpu(), labels.cpu())
    print(f"Testing accuracy of {accuracy.compute().item() * 100:.3f}%")


# ─── Online learning ─────────────────────────────────────────────────────────

def online_learning(encode, model, train_ld, shadow_weight, lr: int = 32):
    """Quantization-aware online update. Returns updated shadow_weight."""
    model.weight = nn.Parameter(shadow_weight.clone().to(device), requires_grad=False)
    all_hv, all_labels = batch_encode(encode, train_ld)
    with torch.no_grad():
        perm = torch.randperm(all_hv.size(0))
        for i in tqdm(range(all_hv.size(0)), desc="Online (QA)"):
            model.add_online_quantized_aware(
                all_hv[perm[i]:perm[i]+1],
                all_labels[perm[i]:perm[i]+1],
                lr,
            )
    return model.weight.detach().clone()


def online_learning_standard(encode, model, train_ld, shadow_weight, lr: float = 16.0):
    """Standard OnlineHD error-corrective update. Returns updated shadow_weight."""
    model.weight = nn.Parameter(shadow_weight.clone().to(device), requires_grad=False)
    all_hv, all_labels = batch_encode(encode, train_ld)
    with torch.no_grad():
        perm = torch.randperm(all_hv.size(0))
        for i in tqdm(range(all_hv.size(0)), desc="Online (std)"):
            model.add_online(
                all_hv[perm[i]:perm[i]+1],
                all_labels[perm[i]:perm[i]+1],
                lr,
                False,
            )
    return model.weight.detach().clone()


# ─── Evaluate one dataset ────────────────────────────────────────────────────

def evaluate_dataset(name: str, dataset_cls, data_root: str, use_standard: bool):
    num_levels = DATASET_LEVELS.get(name, NUM_LEVELS)
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  |  Mode: {'OnlineHD' if use_standard else 'QA'}")
    print(f"  DIMENSIONS={DIMENSIONS}  NUM_LEVELS={num_levels}")
    print(f"{'='*60}")

    train_ld, test_ld, num_features, num_classes = get_loaders(dataset_cls, data_root)
    print(f"  features={num_features}  classes={num_classes}")

    encode = FlatLevelEncoder(num_features, DIMENSIONS, num_levels).to(device)
    model  = Centroid(DIMENSIONS, num_classes).to(device)

    online_fn = online_learning_standard if use_standard else online_learning

    # 1. Train + initial test
    train(encode, model, train_ld)
    # Scale shadow weights by 2 to raise the QA correction threshold,
    # reducing sign-flip instability during online updates on smaller datasets.
    shadow = model.weight.detach().clone()
    test(encode, model, test_ld, num_classes)

    # 2. Online epoch 1
    shadow = online_fn(encode, model, train_ld, shadow)
    test(encode, model, test_ld, num_classes)

    # 3. Online epoch 2
    shadow = online_fn(encode, model, train_ld, shadow)
    test(encode, model, test_ld, num_classes)


# ─── Main ────────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "ISOLET": datasets.ISOLET,
    "UCIHAR": datasets.UCIHAR,
}

# Per-dataset level overrides (tuned for each dataset's feature granularity)
DATASET_LEVELS = {
    "ISOLET": 64,
    "UCIHAR": 32,
}


def main():
    if len(sys.argv) < 2:
        print("Usage: eval_other_datasets.py <ISOLET|UCIHAR> [onlineHD]")
        sys.exit(1)

    dataset_name = sys.argv[1].upper()
    use_standard = len(sys.argv) > 2 and sys.argv[2].lower() == "onlinehd"

    if dataset_name not in DATASET_MAP:
        print(f"Unknown dataset '{dataset_name}'. Choose from: {list(DATASET_MAP)}")
        sys.exit(1)

    evaluate_dataset(
        dataset_name,
        DATASET_MAP[dataset_name],
        str(project_dir / "data"),
        use_standard,
    )


if __name__ == "__main__":
    main()
