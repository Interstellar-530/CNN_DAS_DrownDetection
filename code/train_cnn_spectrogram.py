"""
Train a small CNN on the prepared log-mel spectrogram dataset.

Default task:
  binary_scream: scream vs non_scream, using manifest.csv binary_label.

Useful alternatives:
  three_class: NotScreaming vs Screaming vs normal_speech.
  human_voice: Screaming/normal_speech vs NotScreaming.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


SPLIT_DIRS = {
    "train": "Training",
    "val": "Validation",
    "test": "Test",
}

TASK_CLASS_NAMES = {
    "binary_scream": ["non_scream", "scream"],
    "four_class_water": ["normal_speech", "scream", "water_splash", "other_non_scream"],
    "three_class": ["NotScreaming", "Screaming", "normal_speech"],
    "human_voice": ["non_human_or_other", "human_voice"],
}

WATER_LABELS = {
    "pouring_water",
    "sea_waves",
    "toilet_flush",
    "water_drops",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a CNN on .npy log-mel spectrograms."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset root containing manifest.csv and Training/Validation/Test.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_CLASS_NAMES),
        default="binary_scream",
        help="Label mapping to train.",
    )
    parser.add_argument("--output", type=Path, default=Path("OUTPUT") / "cnn_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--model-v2", action="store_true",
                        help="Use the stronger SpectrogramCNNV2 encoder.")
    parser.add_argument("--model-width", type=int, default=32,
                        help="Base channel width for SpectrogramCNNV2.")
    parser.add_argument("--dual-pool", action="store_true",
                        help="Use adaptive max+avg pooling instead of avg only.")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of weighted cross entropy.")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter.")
    parser.add_argument("--optimize-metric", default="macro_f1",
                        choices=["macro_f1", "scream_f1", "scream_recall", "accuracy", "water_recall", "dual_recall_min", "dual_recall_sum", "dual_f1_min"],
                        help="Metric used for early stopping and scheduler.")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted cross entropy.",
    )
    parser.add_argument(
        "--class-weight-sqrt",
        action="store_true",
        help="Use sqrt inverse-frequency class weights.",
    )
    parser.add_argument(
        "--class-weight-cap",
        type=float,
        default=3.0,
        help="Upper bound for class weights; 0 disables the cap.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Load .npy files on demand instead of caching arrays in RAM.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use light time/frequency masking on training samples.",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Also export best_model_traced.pt for deployment demos.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a saved checkpoint and refresh validation/test metrics without training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path used with --eval-only. Defaults to <run-dir>/best_model.pt.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_root(dataset_arg: Path | None) -> Path:
    if dataset_arg is not None:
        root = dataset_arg.resolve()
        if not (root / "manifest.csv").is_file():
            raise FileNotFoundError(f"manifest.csv not found under {root}")
        return root

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "cnn_spectrogram_dataset_20260815_220143",
        Path.cwd() / "cnn_spectrogram_dataset_20260815_220143",
        Path.cwd().parent / "cnn_spectrogram_dataset_20260815_220143",
    ]
    for candidate in candidates:
        if (candidate / "manifest.csv").is_file():
            return candidate.resolve()

    matches = list(script_dir.parent.glob("**/cnn_spectrogram_dataset_*/manifest.csv"))
    if matches:
        return matches[0].parent.resolve()
    raise FileNotFoundError("Could not locate cnn_spectrogram_dataset_*/manifest.csv")


def basename_from_manifest(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    win_name = PureWindowsPath(value).name
    if win_name and win_name != value:
        return win_name
    return Path(value).name


def label_for_task(row: dict[str, str], task: str) -> str:
    label = row.get("label", "").strip()
    if task == "binary_scream":
        return row.get("binary_label", "").strip() or (
            "scream" if label == "Screaming" else "non_scream"
        )
    if task == "four_class_water":
        mapped_four = row.get("four_class_label", "").strip()
        if mapped_four in TASK_CLASS_NAMES[task]:
            return mapped_four
        if label == "Screaming":
            return "scream"
        if label == "normal_speech":
            return "normal_speech"
        if label in WATER_LABELS:
            return "water_splash"
        return "other_non_scream"
    if task == "three_class":
        return label
    if task == "human_voice":
        return "human_voice" if label in {"Screaming", "normal_speech"} else "non_human_or_other"
    raise ValueError(f"Unknown task: {task}")


def load_manifest_items(
    dataset_root: Path,
    task: str,
    allow_missing: bool = False,
) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
    class_names = list(TASK_CLASS_NAMES[task])
    items = {split: [] for split in SPLIT_DIRS}
    manifest_path = dataset_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "").strip()
            if split not in SPLIT_DIRS:
                continue
            file_name = basename_from_manifest(row.get("npy_path", ""))
            local_path = dataset_root / SPLIT_DIRS[split] / file_name
            if not local_path.is_file():
                original_path = Path(row.get("npy_path", ""))
                if original_path.is_file():
                    local_path = original_path
                else:
                    raise FileNotFoundError(f"Missing spectrogram file: {local_path}")
            mapped_label = label_for_task(row, task)
            if mapped_label not in class_names:
                class_names.append(mapped_label)
            items[split].append(
                {
                    "path": local_path,
                    "label": mapped_label,
                    "raw_label": row.get("label", ""),
                    "binary_label": row.get("binary_label", ""),
                    "source": row.get("source", ""),
                    "source_file": row.get("source_file", ""),
                }
            )
    for split, split_items in items.items():
        if not split_items and not allow_missing:
            raise ValueError(f"No items found for split: {split}")
    return items, class_names


def compute_train_stats(items: list[dict[str, object]]) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    count = 0
    for item in items:
        arr = np.load(item["path"]).astype(np.float64)
        arr = np.nan_to_num(arr, nan=-80.0, posinf=80.0, neginf=-80.0)
        total += float(arr.sum())
        total_sq += float(np.square(arr).sum())
        count += int(arr.size)
    mean = total / max(count, 1)
    variance = max(total_sq / max(count, 1) - mean * mean, 1e-12)
    return float(mean), float(math.sqrt(variance))


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, object]],
        label_to_index: dict[str, int],
        mean: float,
        std: float,
        cache: bool,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.items = items
        self.label_to_index = label_to_index
        self.mean = float(mean)
        self.std = float(std)
        self.cache = cache
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.cached_arrays = None
        if cache:
            self.cached_arrays = [self._read_array(item["path"]) for item in items]

    def __len__(self) -> int:
        return len(self.items)

    def _read_array(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D spectrogram, got {arr.shape}: {path}")
        return np.nan_to_num(arr, nan=-80.0, posinf=80.0, neginf=-80.0)

    def _augment_array(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        if self.rng.random() < 0.50:
            width = int(self.rng.integers(3, 12))
            start = int(self.rng.integers(0, max(1, arr.shape[0] - width)))
            arr[start : start + width, :] = self.mean
        if self.rng.random() < 0.35:
            width = int(self.rng.integers(3, 12))
            start = int(self.rng.integers(0, max(1, arr.shape[1] - width)))
            arr[:, start : start + width] = self.mean
        return arr

    def __getitem__(self, index: int):
        item = self.items[index]
        if self.cached_arrays is None:
            arr = self._read_array(item["path"])
        else:
            arr = self.cached_arrays[index]
        if self.augment:
            arr = self._augment_array(arr)
        arr = (arr.astype(np.float32) - self.mean) / self.std
        x = torch.from_numpy(arr[None, :, :])
        y = self.label_to_index[item["label"]]
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(index, dtype=torch.long)


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 16, pool=True),
            self._block(16, 32, pool=True),
            self._block(32, 64, pool=True),
            self._block(64, 128, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class SpectrogramCNNV2(nn.Module):
    """Stronger encoder with wider channels and optional avg+max pooling.

    Compared with SpectrogramCNN this model keeps more capacity and retains
    both average and peak information, which is useful for short scream bursts.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.25,
        width: int = 32,
        dual_pool: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.width = int(width)
        self.dual_pool = bool(dual_pool)
        self.features = nn.Sequential(
            self._block(1, c1, pool=True),
            self._block(c1, c2, pool=True),
            self._block(c2, c3, pool=True),
            self._block(c3, c4, pool=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1)) if self.dual_pool else None
        pooled_size = c4 * (2 if self.dual_pool else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_size, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        avg = self.avg_pool(x).flatten(1)
        if self.max_pool is not None:
            max_val = self.max_pool(x).flatten(1)
            pooled = torch.cat([avg, max_val], dim=1)
        else:
            pooled = avg
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    """Focal loss for softmax classification."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def make_loader(
    dataset: SpectrogramDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights_for(
    items: list[dict[str, object]],
    class_names: list[str],
    sqrt: bool = False,
    cap: float | None = None,
) -> torch.Tensor:
    counts = Counter(str(item["label"]) for item in items)
    total = sum(counts.values())
    values = []
    for name in class_names:
        weight = total / (len(class_names) * max(counts.get(name, 0), 1))
        if sqrt:
            weight = math.sqrt(weight)
        if cap is not None and cap > 0:
            weight = min(weight, float(cap))
        values.append(weight)
    return torch.tensor(values, dtype=torch.float32)


def metrics_from_logits(
    targets: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    roc_auc = None
    if len(np.unique(targets)) > 1:
        try:
            if len(class_names) == 2:
                roc_auc = float(roc_auc_score(targets, probs[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        targets,
                        probs,
                        labels=list(range(len(class_names))),
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except ValueError:
            roc_auc = None
    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "per_class": per_class,
    }


def metric_score(metrics: dict[str, object], metric_name: str) -> float:
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    if metric_name == "scream_f1":
        return float(metrics.get("per_class", {}).get("scream", {}).get("f1", 0.0))
    if metric_name == "scream_recall":
        return float(metrics.get("per_class", {}).get("scream", {}).get("recall", 0.0))
    if metric_name == "water_recall":
        return float(metrics.get("per_class", {}).get("water_splash", {}).get("recall", 0.0))
    if metric_name in {"dual_recall_min", "dual_recall_sum"}:
        scream_recall = float(metrics.get("per_class", {}).get("scream", {}).get("recall", 0.0))
        water_recall = float(metrics.get("per_class", {}).get("water_splash", {}).get("recall", 0.0))
        return min(scream_recall, water_recall) if metric_name == "dual_recall_min" else scream_recall + water_recall
    if metric_name == "dual_f1_min":
        scream_f1 = float(metrics.get("per_class", {}).get("scream", {}).get("f1", 0.0))
        water_f1 = float(metrics.get("per_class", {}).get("water_splash", {}).get("f1", 0.0))
        return min(scream_f1, water_f1)
    return float(metrics["macro_f1"])


def format_optional_metric(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    all_logits = []
    all_targets = []
    all_indices = []
    for x, y, idx in loader:
        x = x.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())
        all_indices.append(idx.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    indices_np = np.concatenate(all_indices, axis=0)
    return total_loss / max(total_count, 1), {}, logits_np, targets_np, indices_np


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_history(path: Path, history: list[dict[str, object]]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "train_scream_f1",
        "train_roc_auc",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_scream_f1",
        "val_roc_auc",
        "optimize_metric",
        "optimize_score",
        "learning_rate",
        "epoch_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_confusion_matrix(path: Path, targets: np.ndarray, preds: np.ndarray, class_names: list[str]) -> None:
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *class_names])
        for idx, name in enumerate(class_names):
            writer.writerow([name, *[int(v) for v in cm[idx]]])


def save_predictions(
    path: Path,
    dataset: SpectrogramDataset,
    indices: np.ndarray,
    targets: np.ndarray,
    logits: np.ndarray,
    class_names: list[str],
) -> None:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    fieldnames = [
        "file",
        "source",
        "raw_label",
        "target",
        "prediction",
        "correct",
        *[f"prob_{name}" for name in class_names],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, item_index in enumerate(indices):
            item = dataset.items[int(item_index)]
            row = {
                "file": Path(item["path"]).name,
                "source": item.get("source", ""),
                "raw_label": item.get("raw_label", ""),
                "target": class_names[int(targets[row_index])],
                "prediction": class_names[int(preds[row_index])],
                "correct": int(preds[row_index] == targets[row_index]),
            }
            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = f"{float(probs[row_index, class_index]):.8f}"
            writer.writerow(row)


def write_workshop_summary(
    path: Path,
    args: argparse.Namespace,
    dataset_root: Path,
    class_counts: dict[str, Counter],
    best_epoch: int,
    best_val: dict[str, object],
    test_metrics: dict[str, object],
    run_dir: Path,
) -> None:
    artifact_base = Path("OUTPUT") / "cnn_runs" / run_dir.name
    lines = [
        "# CNN Training Summary",
        "",
        f"- Task: `{args.task}`",
        f"- Dataset: `{dataset_root.name}`",
        f"- Input: 128 x 128 log-mel spectrogram `.npy` files",
        f"- Model: 4-block 2D CNN with batch normalization and global average pooling",
        f"- Best epoch: {best_epoch}",
        f"- Validation accuracy / macro-F1 / ROC-AUC: {best_val['accuracy']:.4f} / {best_val['macro_f1']:.4f} / {format_optional_metric(best_val.get('roc_auc'))}",
        f"- Test accuracy / macro-F1 / ROC-AUC: {test_metrics['accuracy']:.4f} / {test_metrics['macro_f1']:.4f} / {format_optional_metric(test_metrics.get('roc_auc'))}",
        "",
        "## Split Counts",
        "",
    ]
    for split in ["train", "val", "test"]:
        counts = class_counts[split]
        joined = ", ".join(f"{name}: {counts.get(name, 0)}" for name in TASK_CLASS_NAMES[args.task])
        lines.append(f"- {split}: {joined}")
    lines.extend(
        [
            "",
            "## Test Per-Class Metrics",
            "",
        ]
    )
    for name, metrics in test_metrics["per_class"].items():
        lines.append(
            "- {name}: precision={precision:.4f}, recall={recall:.4f}, "
            "F1={f1:.4f}, support={support}".format(name=name, **metrics)
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Best model checkpoint: `{artifact_base / 'best_model.pt'}`",
            f"- Test predictions: `{artifact_base / 'test_predictions.csv'}`",
            f"- Confusion matrix: `{artifact_base / 'test_confusion_matrix.csv'}`",
            f"- Full metrics: `{artifact_base / 'metrics_summary.json'}`",
            "",
            "## Notes For Workshop",
            "",
            "- COMSOL simulation was intentionally not included in this run.",
            "- The current split is random within the prepared public datasets; for field deployment, collect local DAS/playback samples and keep an external test set.",
        ]
    )
    if args.task == "four_class_water":
        lines.append(
            "- This four-class run maps ESC-50 water-related clips to `water_splash`; it is a public-data baseline, not a replacement for real DAS water-entry/splash collection."
        )
    elif args.task == "binary_scream":
        lines.append(
            "- The binary run detects `Screaming` vs `non_scream`; tune the decision threshold if the demo prioritizes scream recall over precision."
        )
    elif args.task == "human_voice":
        lines.append(
            "- The human-voice run detects voice-like audio vs non-voice audio and should not be presented as a rescue-state detector by itself."
        )
    else:
        lines.append(
            "- This run is an audio-event classification baseline; the 15-30 second `need_rescue` temporal rule still requires continuous time-series windows."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_root = resolve_dataset_root(args.dataset)
    items_by_split, class_names = load_manifest_items(
        dataset_root, args.task, allow_missing=args.eval_only
    )
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_label = {idx: name for name, idx in label_to_index.items()}

    run_name = args.run_name or f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = (args.output / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.eval_only:
        checkpoint_path = (args.checkpoint or (run_dir / "best_model.pt")).resolve()
        checkpoint = load_checkpoint(checkpoint_path, device)
        mean = float(checkpoint.get("train_mean", -14.238170409173696))
        std = float(checkpoint.get("train_std", 21.23290543312406))
        if std <= 0:
            std = 1.0
        model_class = str(checkpoint.get("model_class", "SpectrogramCNN"))
        model_config = checkpoint.get("model_config", {}) or {}
        if model_class == "SpectrogramCNNV2":
            model = SpectrogramCNNV2(
                num_classes=len(class_names),
                dropout=float(model_config.get("dropout", args.dropout)),
                width=int(model_config.get("width", args.model_width)),
                dual_pool=bool(model_config.get("dual_pool", True)),
            ).to(device)
        else:
            model = SpectrogramCNN(
                num_classes=len(class_names),
                dropout=float(model_config.get("dropout", args.dropout)),
            ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        weights = None
        criterion = nn.CrossEntropyLoss()
    else:
        mean, std = compute_train_stats(items_by_split["train"])
        if args.model_v2:
            model = SpectrogramCNNV2(
                num_classes=len(class_names),
                dropout=args.dropout,
                width=args.model_width,
                dual_pool=args.dual_pool,
            ).to(device)
            model_class = "SpectrogramCNNV2"
            model_config = {
                "width": args.model_width,
                "dual_pool": args.dual_pool,
                "dropout": args.dropout,
            }
        else:
            model = SpectrogramCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
            model_class = "SpectrogramCNN"
            model_config = {"dropout": args.dropout}
        weights = (
            None
            if args.no_class_weights
            else class_weights_for(
                items_by_split["train"],
                class_names,
                sqrt=args.class_weight_sqrt,
                cap=args.class_weight_cap,
            )
        )
        if args.focal_loss:
            criterion = FocalLoss(
                weight=weights.to(device) if weights is not None else None,
                gamma=args.focal_gamma,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weights.to(device) if weights is not None else None
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

    cache = not args.no_cache
    train_ds = SpectrogramDataset(
        items_by_split["train"], label_to_index, mean, std, cache=cache, augment=args.augment, seed=args.seed
    )
    val_ds = SpectrogramDataset(
        items_by_split["val"], label_to_index, mean, std, cache=cache, augment=False, seed=args.seed
    )
    test_ds = SpectrogramDataset(
        items_by_split["test"], label_to_index, mean, std, cache=cache, augment=False, seed=args.seed
    )
    train_loader = (
        make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
        if len(items_by_split["train"]) > 0
        else None
    )
    val_loader = (
        make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
        if len(items_by_split["val"]) > 0
        else None
    )
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_counts = {
        split: Counter(str(item["label"]) for item in split_items)
        for split, split_items in items_by_split.items()
    }
    args_config = vars(args).copy()
    args_config["dataset"] = str(dataset_root)
    args_config["output"] = str(args.output)
    args_config["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
    config = {
        "args": args_config,
        "run_dir": str(run_dir),
        "class_names": class_names,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "train_mean": mean,
        "train_std": std,
        "class_counts": {
            split: {name: int(counts.get(name, 0)) for name in class_names}
            for split, counts in class_counts.items()
        },
        "class_weights": None if weights is None else [float(v) for v in weights.tolist()],
        "model_class": model_class,
        "model_config": model_config,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    save_json(run_dir / ("eval_config.json" if args.eval_only else "run_config.json"), config)

    print(f"Dataset: {dataset_root}")
    print(f"Run dir: {run_dir}")
    print(f"Task/classes: {args.task} -> {class_names}")
    print(f"Device: {device}")
    print(f"Train mean/std: {mean:.4f}/{std:.4f}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {dict(class_counts[split])}")

    if args.eval_only:
        test_loss, _, test_logits, test_targets, test_indices = run_epoch(
            model, test_loader, criterion, device, optimizer=None
        )
        test_metrics = metrics_from_logits(test_targets, test_logits, class_names)
        test_preds = test_logits.argmax(axis=1)
        save_confusion_matrix(run_dir / "test_confusion_matrix.csv", test_targets, test_preds, class_names)
        save_predictions(run_dir / "test_predictions.csv", test_ds, test_indices, test_targets, test_logits, class_names)

        if len(items_by_split["val"]) > 0:
            val_loss, _, val_logits, val_targets, val_indices = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )
            best_val = metrics_from_logits(val_targets, val_logits, class_names)
            val_preds = val_logits.argmax(axis=1)
            save_confusion_matrix(run_dir / "validation_confusion_matrix.csv", val_targets, val_preds, class_names)
            save_predictions(run_dir / "validation_predictions.csv", val_ds, val_indices, val_targets, val_logits, class_names)
            best_validation_loss = float(val_loss)
        else:
            best_val = {"accuracy": 0.0, "macro_f1": 0.0, "roc_auc": None, "per_class": {}}
            best_validation_loss = None

        old_summary_path = run_dir / "metrics_summary.json"
        history_tail = []
        if old_summary_path.is_file():
            try:
                with old_summary_path.open("r", encoding="utf-8") as f:
                    history_tail = json.load(f).get("history_tail", [])
            except Exception:
                history_tail = []
        best_epoch = int(checkpoint.get("epoch", 0))
        metrics_summary = {
            "best_epoch": best_epoch,
            "evaluated_checkpoint": str(checkpoint_path),
            "best_validation_loss": best_validation_loss,
            "best_validation": best_val,
            "test_loss": float(test_loss),
            "test": test_metrics,
            "history_tail": history_tail,
        }
        save_json(run_dir / "metrics_summary.json", metrics_summary)
        write_workshop_summary(
            run_dir / "workshop_cnn_summary.md",
            args,
            dataset_root,
            class_counts,
            best_epoch,
            best_val,
            test_metrics,
            run_dir,
        )

        print("\nDone.")
        print(f"Evaluated checkpoint: {checkpoint_path}")
        print(f"Best epoch: {best_epoch}")
        print(
            "Validation acc/macro-F1/ROC-AUC: "
            f"{best_val.get('accuracy', 0.0):.4f}/{best_val.get('macro_f1', 0.0):.4f}/"
            f"{format_optional_metric(best_val.get('roc_auc'))}"
        )
        print(
            "Test acc/macro-F1/ROC-AUC: "
            f"{test_metrics['accuracy']:.4f}/{test_metrics['macro_f1']:.4f}/{format_optional_metric(test_metrics.get('roc_auc'))}"
        )
        print(f"Saved: {run_dir}")
        return

    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, train_logits, train_targets, _ = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        train_metrics = metrics_from_logits(train_targets, train_logits, class_names)
        val_loss, _, val_logits, val_targets, _ = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        val_metrics = metrics_from_logits(val_targets, val_logits, class_names)
        scheduler.step(metric_score(val_metrics, args.optimize_metric))
        epoch_seconds = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_scream_f1": float(train_metrics["per_class"].get("scream", {}).get("f1", 0.0)),
            "train_roc_auc": train_metrics["roc_auc"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_scream_f1": float(val_metrics["per_class"].get("scream", {}).get("f1", 0.0)),
            "val_roc_auc": val_metrics["roc_auc"],
            "optimize_metric": args.optimize_metric,
            "optimize_score": metric_score(val_metrics, args.optimize_metric),
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_metrics['accuracy']:.4f} "
            f"f1 {train_metrics['macro_f1']:.4f} auc {format_optional_metric(train_metrics.get('roc_auc'))} "
            f"| val loss {val_loss:.4f} acc {val_metrics['accuracy']:.4f} "
            f"f1 {val_metrics['macro_f1']:.4f} auc {format_optional_metric(val_metrics.get('roc_auc'))} "
            f"scream_f1 {float(val_metrics['per_class'].get('scream', {}).get('f1', 0.0)):.4f} "
            f"| lr {current_lr:.2e} | {epoch_seconds:.1f}s"
        )

        score = metric_score(val_metrics, args.optimize_metric)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "class_names": class_names,
                "label_to_index": label_to_index,
                "task": args.task,
                "train_mean": mean,
                "train_std": std,
                "input_shape": [1, 128, 128],
                "model_class": model_class,
                "model_config": model_config,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": config,
            }
            torch.save(checkpoint, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {bad_epochs} epochs without validation improvement.")
                break

    save_history(run_dir / "history.csv", history)

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    val_loss, _, val_logits, val_targets, val_indices = run_epoch(
        model, val_loader, criterion, device, optimizer=None
    )
    test_loss, _, test_logits, test_targets, test_indices = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )
    best_val = metrics_from_logits(val_targets, val_logits, class_names)
    test_metrics = metrics_from_logits(test_targets, test_logits, class_names)
    val_preds = val_logits.argmax(axis=1)
    test_preds = test_logits.argmax(axis=1)
    save_confusion_matrix(run_dir / "validation_confusion_matrix.csv", val_targets, val_preds, class_names)
    save_predictions(run_dir / "validation_predictions.csv", val_ds, val_indices, val_targets, val_logits, class_names)
    save_confusion_matrix(run_dir / "test_confusion_matrix.csv", test_targets, test_preds, class_names)
    save_predictions(run_dir / "test_predictions.csv", test_ds, test_indices, test_targets, test_logits, class_names)

    best_epoch = int(checkpoint.get("epoch", best_epoch))
    metrics_summary = {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(val_loss),
        "best_validation": best_val,
        "test_loss": float(test_loss),
        "test": test_metrics,
        "history_tail": history[-5:],
    }
    save_json(run_dir / "metrics_summary.json", metrics_summary)

    if args.export_torchscript:
        model_cpu = model.to("cpu").eval()
        with torch.no_grad():
            traced = torch.jit.trace(model_cpu, torch.zeros(1, 1, 128, 128))
        with (run_dir / "best_model_traced.pt").open("wb") as f:
            torch.jit.save(traced, f)
        model.to(device)

    write_workshop_summary(
        run_dir / "workshop_cnn_summary.md",
        args,
        dataset_root,
        class_counts,
        best_epoch,
        best_val,
        test_metrics,
        run_dir,
    )

    print("\nDone.")
    print(f"Best epoch: {best_epoch}")
    print(
        "Validation acc/macro-F1/ROC-AUC: "
        f"{best_val['accuracy']:.4f}/{best_val['macro_f1']:.4f}/{format_optional_metric(best_val.get('roc_auc'))}"
    )
    print(
        "Validation scream F1: "
        f"{float(best_val['per_class'].get('scream', {}).get('f1', 0.0)):.4f}"
    )
    print(
        "Test acc/macro-F1/ROC-AUC: "
        f"{test_metrics['accuracy']:.4f}/{test_metrics['macro_f1']:.4f}/{format_optional_metric(test_metrics.get('roc_auc'))}"
    )
    print(
        "Test scream F1: "
        f"{float(test_metrics['per_class'].get('scream', {}).get('f1', 0.0)):.4f}"
    )
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
