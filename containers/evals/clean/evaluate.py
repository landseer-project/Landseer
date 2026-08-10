#!/usr/bin/env python3
"""Clean accuracy evaluator: train/test (+ poisoned when metadata present) and mia_auc."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_loader import load_torch_model_for_eval


def evaluate_clean_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predicted = model(images).argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total > 0 else 0.0


def _make_loader(x, y, batch_size):
    dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _read_dp_metrics(input_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    metrics_file = input_dir / "privacy_metrics.txt"
    if not metrics_file.exists():
        return None, None, None
    epsilon = dp_accuracy = mia_auc = None
    try:
        for raw_line in metrics_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key == "epsilon":
                epsilon = float(value)
            elif key in ("dp_accuracy", "best_accuracy"):
                dp_accuracy = float(value)
            elif key == "mia_auc":
                mia_auc = float(value)
    except Exception:
        return None, None, None
    return epsilon, dp_accuracy, mia_auc


def _normalize_accuracy(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _apply_badnets_trigger(images, trigger_info):
    triggered = images.copy()
    size = int(trigger_info.get("trigger_size", 3))
    value = float(trigger_info.get("trigger_value", 1.0))
    pos = trigger_info.get("trigger_position", "bottom_right")
    _, _, h, w = triggered.shape
    if pos == "top_left":
        triggered[:, :, :size, :size] = value
    elif pos == "top_right":
        triggered[:, :, :size, w - size :] = value
    elif pos == "bottom_left":
        triggered[:, :, h - size :, :size] = value
    else:
        triggered[:, :, h - size :, w - size :] = value
    return triggered


def _mia_auc(model, train_loader, test_loader, device):
    """Loss-based membership inference AUC (train=member)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    model.eval()
    scores, labels = [], []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for loader, is_member in ((train_loader, 1), (test_loader, 0)):
            for images, y in loader:
                images = images.to(device)
                y = y.to(device)
                losses = loss_fn(model(images), y).detach().cpu().numpy()
                # lower loss => more likely member
                scores.extend((-losses).tolist())
                labels.extend([is_member] * len(losses))
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def write_results(output_dir, payload):
    (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    config = {}
    config_path = input_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    batch_size = config.get("batch_size", 128)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = input_dir / "model.pt"
    test_data_path = input_dir / "test_data.npy"
    test_labels_path = input_dir / "test_labels.npy"
    train_data_path = input_dir / "data.npy"
    train_labels_path = input_dir / "labels.npy"

    if not model_path.exists():
        write_results(output_dir, {"evaluator": "clean", "success": False, "error": "model.pt not found", "metrics": {}})
        return
    if not test_data_path.exists() or not test_labels_path.exists():
        write_results(output_dir, {"evaluator": "clean", "success": False, "error": "Test data not found", "metrics": {}})
        return

    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
    except Exception as exc:
        write_results(output_dir, {"evaluator": "clean", "success": False, "error": f"Failed to load model: {exc}", "metrics": {}})
        return

    x_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)
    test_loader = _make_loader(x_test, y_test, batch_size)
    clean_test_acc = evaluate_clean_accuracy(model, test_loader, device)

    metrics = {}
    train_loader = None
    clean_train_acc = None
    if train_data_path.exists() and train_labels_path.exists():
        x_train = np.load(train_data_path)
        y_train = np.load(train_labels_path)
        train_loader = _make_loader(x_train, y_train, batch_size)
        clean_train_acc = evaluate_clean_accuracy(model, train_loader, device)
        metrics["clean_train_accuracy"] = float(clean_train_acc)
        print(f"Clean train accuracy: {clean_train_acc:.4f}")

    epsilon, dp_accuracy, mia_from_file = _read_dp_metrics(input_dir)
    reported_clean = _normalize_accuracy(dp_accuracy) if dp_accuracy is not None else clean_test_acc
    metrics["clean_accuracy"] = float(reported_clean)
    if dp_accuracy is not None:
        metrics["raw_clean_accuracy"] = float(clean_test_acc)
        metrics["dp_accuracy"] = float(dp_accuracy)
    if epsilon is not None:
        metrics["privacy_epsilon"] = float(epsilon)
    print(f"Clean test accuracy: {reported_clean:.4f}")

    # Poisoned accuracies when backdoor metadata is present
    meta_path = input_dir / "poisoning_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            trigger_info = meta.get("trigger_info") or meta
            if x_test.ndim == 4:
                x_poison = _apply_badnets_trigger(x_test.astype(np.float32), trigger_info)
                poison_loader = _make_loader(x_poison, y_test, batch_size)
                poisoned_test_acc = evaluate_clean_accuracy(model, poison_loader, device)
                metrics["poisoned_test_accuracy"] = float(poisoned_test_acc)
                print(f"Poisoned test accuracy: {poisoned_test_acc:.4f}")
            if train_loader is not None:
                # Train set is already poisoned for poisoned pipelines
                metrics["poisoned_train_accuracy"] = float(clean_train_acc)
        except Exception as exc:
            print(f"Poisoned accuracy skipped: {exc}")

    # MIA AUC (file override, else loss-based)
    mia_auc = mia_from_file
    if mia_auc is None and train_loader is not None:
        mia_auc = _mia_auc(model, train_loader, test_loader, device)
    if mia_auc is not None:
        metrics["mia_auc"] = float(mia_auc if mia_auc <= 1.0 else mia_auc / 100.0)
        print(f"MIA AUC: {metrics['mia_auc']:.4f}")

    write_results(
        output_dir,
        {
            "evaluator": "clean",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {"batch_size": batch_size, "device": device},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


if __name__ == "__main__":
    main()
