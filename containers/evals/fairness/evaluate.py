#!/usr/bin/env python3
"""
Fairness Evaluator

Evaluates model fairness using:
- Demographic Parity Difference
- Equalized Odds Difference
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_fairness(model, test_X, test_y, sensitive_attrs, device, method="fairlearn"):
    """
    Evaluate fairness metrics.
    
    Args:
        model: Trained model
        test_X: Test features
        test_y: Test labels
        sensitive_attrs: Sensitive attributes
        device: Device to use
        method: 'fairlearn' or 'custom'
        
    Returns:
        Tuple of (demographic_parity, equalized_odds_diff)
    """
    model.eval()
    preds = []
    
    test_loader = DataLoader(
        TensorDataset(test_X, test_y),
        batch_size=64
    )
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.extend(outputs.argmax(1).cpu().numpy())
    
    y_true = test_y.cpu().numpy()
    sensitive_attrs = np.array(sensitive_attrs)
    preds = np.array(preds)
    
    if method == "fairlearn":
        from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
        
        dp = demographic_parity_difference(
            y_true=y_true,
            y_pred=preds,
            sensitive_features=sensitive_attrs
        )
        deo = equalized_odds_difference(
            y_true=y_true,
            y_pred=preds,
            sensitive_features=sensitive_attrs
        )
    else:
        # Custom implementation
        dp = abs(
            np.mean([p for p, z in zip(preds, sensitive_attrs) if z == 1]) -
            np.mean([p for p, z in zip(preds, sensitive_attrs) if z == 0])
        )
        
        # TPR for each group
        tpr_z1 = np.mean([
            1 if p == 1 else 0
            for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 1 and z == 1
        ]) if np.sum((y_true == 1) & (sensitive_attrs == 1)) > 0 else 0
        
        tpr_z0 = np.mean([
            1 if p == 1 else 0
            for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 1 and z == 0
        ]) if np.sum((y_true == 1) & (sensitive_attrs == 0)) > 0 else 0
        
        # FPR for each group
        fpr_z1 = np.mean([
            1 if p == 1 else 0
            for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 0 and z == 1
        ]) if np.sum((y_true == 0) & (sensitive_attrs == 1)) > 0 else 0
        
        fpr_z0 = np.mean([
            1 if p == 1 else 0
            for p, y, z in zip(preds, y_true, sensitive_attrs)
            if y == 0 and z == 0
        ]) if np.sum((y_true == 0) & (sensitive_attrs == 0)) > 0 else 0
        
        deo = abs(tpr_z1 - tpr_z0) + abs(fpr_z1 - fpr_z0)
    
    return dp, deo


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load config
    config_path = input_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    
    method = config.get("method", "fairlearn")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for required sensitive attributes
    sensitive_path = input_dir / "sensitive_attributes.npy"
    if not sensitive_path.exists():
        # Graceful skip
        results = {
            "evaluator": "fairness",
            "success": True,
            "skipped": True,
            "skip_reason": "sensitive_attributes.npy not found",
            "metrics": {
                "demographic_parity": None,
                "equalized_odds_diff": None
            }
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        print("Skipping: sensitive_attributes.npy not found")
        return
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "fairness",
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
            "evaluator": "fairness",
            "success": False,
            "error": f"Failed to load model: {e}",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    # Load test data
    X = np.load(input_dir / "test_data.npy")
    Y = np.load(input_dir / "test_labels.npy")
    sensitive_attrs = np.load(sensitive_path)
    
    test_X = torch.tensor(X).float()
    test_y = torch.tensor(Y).long()
    
    print(f"Loaded {len(test_X)} test samples with sensitive attributes")
    
    # Evaluate
    try:
        dp, deo = evaluate_fairness(
            model, test_X, test_y, sensitive_attrs, device, method
        )
        
        metrics = {
            "demographic_parity": float(dp),
            "equalized_odds_diff": float(deo)
        }
        
        print(f"Demographic Parity: {dp:.4f}")
        print(f"Equalized Odds Difference: {deo:.4f}")
        
        results = {
            "evaluator": "fairness",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {"method": method},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        results = {
            "evaluator": "fairness",
            "success": False,
            "error": str(e),
            "metrics": {}
        }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
