#!/usr/bin/env python3
"""Benchmark QA online-learning across learning rates, dimensions, and datasets.

Sweeps:
  - Datasets  : ISOLET (64 levels), UCIHAR (32 levels)
  - Dimensions: 1 000, 4 000, 10 000
  - LR values : 10 → 100 in steps of 10
  - Runs      : 20 per (dataset × dim × lr)

Output files (written incrementally):
  results_ISOLET_lr_sweep.txt
  results_UCIHAR_lr_sweep.txt

Format:
  === DIMENSIONS 1000 ===
  --- LR 10 ---
  test: XX.XXX epoch1: XX.XXX epoch2: XX.XXX
  ...
  --- LR 20 ---
  ...
"""

import pathlib
import sys
import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.transforms as tvt

import torchhd
import torchhd.datasets as datasets

script_dir = pathlib.Path(__file__).parent.resolve()
repo_root = script_dir.parent
project_dir = repo_root / "mnist_example"

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from quant_models import Centroid

# ─── Config ──────────────────────────────────────────────────────────────────

DIMENSIONS_LIST = [1_000, 4_000, 10_000]
#DIMENSIONS_LIST = [10_000]
#LR_VALUES       = list(range(10, 101, 10))   # [10, 20, ..., 100]
LR_VALUES       = list(range(180, 201, 20))   # [10, 20, ..., 100]
RUNS            = 20
BATCH_SIZE      = 32

# Keep MNIST pixels as uint8 [0, 255] — matches the PILToTensor transform used in hdc.py
_mnist_transform = tvt.PILToTensor()

# Per-dataset config: dataset class, level embedding size, optional transform, encoder class
DATASET_CONFIG = {
    "MNIST":  {"cls": torchvision.datasets.MNIST,  "levels": 256, "transform": _mnist_transform, "encoder_cls": "MNISTEncoder"},
    #"ISOLET": {"cls": datasets.ISOLET, "levels": 64,  "transform": None, "encoder_cls": "FlatLevelEncoder"},
    #"UCIHAR": {"cls": datasets.UCIHAR, "levels": 32,  "transform": None, "encoder_cls": "FlatLevelEncoder"},
}

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Encoders ────────────────────────────────────────────────────────────────

class FlatLevelEncoder(nn.Module):
    """Position × level binding encoder for flat real-valued feature vectors."""

    def __init__(self, in_features: int, out_features: int, levels: int):
        super().__init__()
        self.position = torchhd.embeddings.Random(in_features, out_features)
        self.value    = torchhd.embeddings.Level(levels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


class MNISTEncoder(nn.Module):
    """Mirrors BaseLevelEncoder from hdc.py: flattens uint8 image, then bind+multiset.

    Accepts uint8 tensors (PILToTensor output) directly — no pre-normalisation needed.
    """

    def __init__(self, in_features: int, out_features: int, levels: int):
        super().__init__()
        self.flatten  = nn.Flatten()
        self.position = torchhd.embeddings.Random(in_features, out_features)
        self.value    = torchhd.embeddings.Level(levels, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x         = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x)).to(x.device)
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


ENCODER_CLASSES = {
    "FlatLevelEncoder": FlatLevelEncoder,
    "MNISTEncoder":     MNISTEncoder,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def get_loaders(dataset_cls, data_root: str, transform=None):
    kwargs = {"transform": transform} if transform is not None else {}
    train_ds = dataset_cls(data_root, train=True,  download=True, **kwargs)
    test_ds  = dataset_cls(data_root, train=False, download=True, **kwargs)

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


# ─── Train / eval helpers ─────────────────────────────────────────────────────

def batch_encode(encode: nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
    all_hv, all_labels = [], []
    with torch.no_grad():
        for samples, labels in loader:
            all_hv.append(encode(samples.float().to(device)))
            all_labels.append(labels.to(device))
    return torch.cat(all_hv), torch.cat(all_labels)


def train(encode: nn.Module, model: Centroid, train_ld) -> None:
    with torch.no_grad():
        for samples, labels in train_ld:
            model.add(encode(samples.float().to(device)), labels.to(device))


def eval_accuracy(encode: nn.Module, model: Centroid, test_ld, num_classes: int) -> float:
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    model.normalize(quantize=True)
    with torch.no_grad():
        for samples, labels in test_ld:
            outputs = model(encode(samples.float().to(device)), dot=True)
            accuracy.update(outputs.cpu(), labels.cpu())
    return accuracy.compute().item() * 100.0


def online_qa(
    encode: nn.Module,
    model: Centroid,
    train_ld,
    shadow_weight: torch.Tensor,
    lr: int,
) -> torch.Tensor:
    """One QA online-learning epoch. Returns updated shadow weight."""
    model.weight = nn.Parameter(shadow_weight.clone().to(device), requires_grad=False)
    all_hv, all_labels = batch_encode(encode, train_ld)
    perm = torch.randperm(all_hv.size(0))
    with torch.no_grad():
        for i in range(all_hv.size(0)):
            model.add_online_quantized_aware(
                all_hv[perm[i] : perm[i] + 1],
                all_labels[perm[i] : perm[i] + 1],
                lr,
            )
    return model.weight.detach().clone()


# ─── Single run ───────────────────────────────────────────────────────────────

def run_once(
    dataset_cls,
    data_root: str,
    dimensions: int,
    num_levels: int,
    lr: int,
    transform=None,
    encoder_cls: str = "FlatLevelEncoder",
) -> tuple[float, float, float]:
    train_ld, test_ld, num_features, num_classes = get_loaders(dataset_cls, data_root, transform)

    encode = ENCODER_CLASSES[encoder_cls](num_features, dimensions, num_levels).to(device)
    model  = Centroid(dimensions, num_classes).to(device)

    # Initial training pass
    train(encode, model, train_ld)
    shadow = model.weight.detach().clone()
    acc0   = eval_accuracy(encode, model, test_ld, num_classes)

    # Online epoch 1
    shadow = online_qa(encode, model, train_ld, shadow, lr)
    acc1   = eval_accuracy(encode, model, test_ld, num_classes)

    # Online epoch 2
    shadow = online_qa(encode, model, train_ld, shadow, lr)
    acc2   = eval_accuracy(encode, model, test_ld, num_classes)

    return acc0, acc1, acc2


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    data_root = str(project_dir / "data")

    for ds_name, ds_cfg in DATASET_CONFIG.items():
        out_path = project_dir / f"results_{ds_name}_lr_sweep.txt"

        print(f"\n{'='*60}")
        print(f"  Dataset : {ds_name}")
        print(f"  Levels  : {ds_cfg['levels']}")
        print(f"  Dims    : {DIMENSIONS_LIST}")
        print(f"  LR range: {LR_VALUES[0]}–{LR_VALUES[-1]} (step 10)")
        print(f"  Runs    : {RUNS}")
        print(f"  Output  : {out_path}")
        print(f"{'='*60}")

        with open(out_path, "w") as out:
            for dim in DIMENSIONS_LIST:
                out.write(f"=== DIMENSIONS {dim} ===\n")
                out.flush()
                print(f"\n  [dim={dim}]")

                for lr in LR_VALUES:
                    out.write(f"--- LR {lr} ---\n")
                    out.flush()
                    print(f"    LR={lr}")

                    for run in range(1, RUNS + 1):
                        print(f"      run {run:2d}/{RUNS} …", end=" ", flush=True)
                        try:
                            acc0, acc1, acc2 = run_once(
                                ds_cfg["cls"],
                                data_root,
                                dim,
                                ds_cfg["levels"],
                                lr,
                                ds_cfg.get("transform"),
                                ds_cfg.get("encoder_cls", "FlatLevelEncoder"),
                            )
                            line = (
                                f"test: {acc0:.3f} "
                                f"epoch1: {acc1:.3f} "
                                f"epoch2: {acc2:.3f}"
                            )
                        except Exception as exc:
                            line = f"ERROR: {exc}"

                        out.write(line + "\n")
                        out.flush()
                        print(line)

        print(f"\n  Done -> {out_path}")

    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()
