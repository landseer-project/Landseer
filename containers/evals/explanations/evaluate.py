#!/usr/bin/env python3
"""Explanation fidelity: Drop10 (accuracy drop after zeroing top-10% gradient pixels)."""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model_loader import load_torch_model_for_eval


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def drop10_score(model, x, y, device, frac=0.1, max_samples=128):
    """Mean accuracy drop after zeroing top-frac |grad| pixels per sample."""
    n = min(len(x), max_samples)
    x = torch.tensor(x[:n]).float().to(device)
    y = torch.tensor(y[:n]).long().to(device)

    model.eval()
    x_req = x.clone().detach().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(model(x_req), y)
    loss.backward()
    grads = x_req.grad.detach().abs()

    flat = grads.view(n, -1)
    k = max(1, int(flat.shape[1] * frac))
    topk = torch.topk(flat, k, dim=1).indices
    mask = torch.ones_like(flat)
    mask.scatter_(1, topk, 0.0)
    x_drop = (x.view(n, -1) * mask).view_as(x)

    loader_clean = DataLoader(TensorDataset(x.detach().cpu(), y.cpu()), batch_size=32)
    loader_drop = DataLoader(TensorDataset(x_drop.detach().cpu(), y.cpu()), batch_size=32)
    acc_clean = _accuracy(model, loader_clean, device)
    acc_drop = _accuracy(model, loader_drop, device)
    return float(acc_clean - acc_drop)


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    model_path = input_dir / "model.pt"
    test_data = input_dir / "test_data.npy"
    test_labels = input_dir / "test_labels.npy"
    if not model_path.exists() or not test_data.exists() or not test_labels.exists():
        payload = {
            "evaluator": "explanations",
            "success": True,
            "skipped": True,
            "skip_reason": "model.pt or test data missing",
            "metrics": {"drop10_score": None},
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
        score = drop10_score(model, np.load(test_data), np.load(test_labels), device)
        payload = {
            "evaluator": "explanations",
            "success": True,
            "skipped": False,
            "metrics": {"drop10_score": score},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        print(f"drop10_score: {score:.4f}")
    except Exception as exc:
        payload = {
            "evaluator": "explanations",
            "success": False,
            "error": str(exc),
            "metrics": {},
        }

    (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
