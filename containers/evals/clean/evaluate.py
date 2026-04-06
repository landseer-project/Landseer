#!/usr/bin/env python3
"""
Clean accuracy evaluator.

Always runs and reports top-1 accuracy on clean test data.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_loader import load_torch_model_for_eval


def evaluate_clean_accuracy(model, loader, device):
    """Compute top-1 accuracy on clean samples."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


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
    print(f"Using device: {device}")

    model_path = input_dir / "model.pt"
    test_data_path = input_dir / "test_data.npy"
    test_labels_path = input_dir / "test_labels.npy"

    if not model_path.exists():
        write_results(
            output_dir,
            {
                "evaluator": "clean",
                "success": False,
                "error": "model.pt not found",
                "metrics": {},
            },
        )
        return

    if not test_data_path.exists() or not test_labels_path.exists():
        write_results(
            output_dir,
            {
                "evaluator": "clean",
                "success": False,
                "error": "Test data not found",
                "metrics": {},
            },
        )
        return

    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
    except Exception as exc:
        write_results(
            output_dir,
            {
                "evaluator": "clean",
                "success": False,
                "error": f"Failed to load model: {exc}",
                "metrics": {},
            },
        )
        return

    x_test = np.load(test_data_path)
    y_test = np.load(test_labels_path)

    dataset = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    clean_acc = evaluate_clean_accuracy(model, loader, device)
    print(f"Clean accuracy: {clean_acc:.4f}")

    write_results(
        output_dir,
        {
            "evaluator": "clean",
            "success": True,
            "skipped": False,
            "metrics": {"clean_accuracy": float(clean_acc)},
            "parameters": {"batch_size": batch_size, "device": device},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
