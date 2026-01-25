#!/usr/bin/env python3
"""
Fingerprinting Evaluator

Evaluates model fingerprinting using MINGD attack.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def loss_mingd(preds, target):
    """MINGD loss function."""
    return (preds.max(dim=1)[0] - preds[torch.arange(preds.shape[0], device=preds.device), target]).mean()


def mingd(model, X, y, target, alpha=0.01, num_iter=20):
    """
    MINGD attack for fingerprinting evaluation.
    """
    model.eval()
    delta = torch.zeros_like(X, requires_grad=True)
    
    for _ in range(num_iter):
        preds = model(X + delta)
        loss = -loss_mingd(preds, target)
        loss.backward()
        delta.data += alpha * delta.grad.sign()
        delta.data = torch.min(torch.max(delta, -X), 1 - X)
        delta.grad.zero_()
    
    return delta.detach()


def evaluate_fingerprinting_mingd(model, dataloader, device):
    """
    Evaluate fingerprinting using MINGD attack.
    """
    model = model.to(device)
    model.eval()
    
    total = 0
    correct_clean = 0
    correct_mingd = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        total += X.size(0)
        
        # Clean prediction
        with torch.no_grad():
            pred_clean = model(X).max(1)[1]
            correct_clean += (pred_clean == y).sum().item()
        
        # MINGD attack: target is original clean prediction
        delta = mingd(model, X, y, pred_clean)
        
        with torch.no_grad():
            pred_mingd = model(X + delta).max(1)[1]
            correct_mingd += (pred_mingd == y).sum().item()
    
    clean_acc = correct_clean / total if total > 0 else 0.0
    mingd_score = correct_mingd / total if total > 0 else 0.0
    
    return clean_acc, mingd_score


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config_path = input_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    
    batch_size = config.get("batch_size", 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "fingerprinting",
            "success": False,
            "error": "model.pt not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        results = {
            "evaluator": "fingerprinting",
            "success": False,
            "error": f"Failed to load model: {e}",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    # Load test data
    test_data_path = input_dir / "test_data.npy"
    test_labels_path = input_dir / "test_labels.npy"
    
    if not test_data_path.exists() or not test_labels_path.exists():
        results = {
            "evaluator": "fingerprinting",
            "success": False,
            "error": "Test data not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    X = np.load(test_data_path)
    Y = np.load(test_labels_path)
    
    dataset = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(Y).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded {len(dataset)} test samples")
    print(f"Evaluating fingerprinting with MINGD attack...")
    
    try:
        clean_acc, mingd_score = evaluate_fingerprinting_mingd(model, loader, device)
        
        metrics = {
            "clean_accuracy": clean_acc,
            "mingd_score": mingd_score,
            "fingerprint_accuracy": mingd_score
        }
        
        print(f"Clean accuracy: {clean_acc:.4f}")
        print(f"MINGD score: {mingd_score:.4f}")
        
        results = {
            "evaluator": "fingerprinting",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        results = {
            "evaluator": "fingerprinting",
            "success": False,
            "error": str(e),
            "metrics": {}
        }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
