#!/usr/bin/env python3
"""
Out-of-Distribution (OOD) Detection Evaluator

Evaluates model's ability to detect out-of-distribution samples using:
- AUC-ROC score
- FPR at 95% TPR

This matches the original functionality from src_old/landseer_pipeline/evaluator/model_evaluator.py

The original uses:
- Score = 1 - max_confidence (so higher score = more likely OOD)
- CIFAR-100 as OOD dataset for CIFAR-10 models
- Can also use synthetic OOD (Gaussian noise, etc.)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve


def get_max_confidence_scores(model, dataloader, device):
    """
    Get maximum softmax probability scores for samples.
    Returns scores as (1 - max_conf) so higher = more likely OOD.
    This matches the original implementation in model_evaluator.py.
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_conf = probs.max(dim=1)[0]
            # Original uses (1 - max_conf) so higher score = more OOD
            scores.extend((1 - max_conf).cpu().numpy())
    
    return np.array(scores)


def generate_ood_samples(in_distribution_data, method="gaussian"):
    """
    Generate out-of-distribution samples for testing.
    """
    n_samples = min(len(in_distribution_data), 2000)  # Cap at 2000 samples
    shape = in_distribution_data.shape[1:]
    
    if method == "gaussian":
        # Random Gaussian noise, normalized to [0, 1]
        ood = np.random.randn(n_samples, *shape).astype(np.float32)
        ood = (ood - ood.min()) / (ood.max() - ood.min() + 1e-8)
    
    elif method == "uniform":
        # Uniform random in [0, 1]
        ood = np.random.rand(n_samples, *shape).astype(np.float32)
    
    elif method == "permuted":
        # Permuted in-distribution samples (shuffled pixels)
        indices = np.random.choice(len(in_distribution_data), n_samples, replace=False)
        ood = in_distribution_data[indices].copy()
        for i in range(len(ood)):
            flat = ood[i].flatten()
            np.random.shuffle(flat)
            ood[i] = flat.reshape(shape)
    
    else:
        ood = np.random.randn(n_samples, *shape).astype(np.float32)
        ood = (ood - ood.min()) / (ood.max() - ood.min() + 1e-8)
    
    return ood.astype(np.float32)


def load_cifar100_ood(transform_normalize=True):
    """
    Load CIFAR-100 as OOD dataset (original method from src_old).
    Returns None if download fails or torchvision unavailable.
    """
    try:
        import torchvision
        from torchvision import transforms
        
        # CIFAR-10 normalization (used for consistency)
        if transform_normalize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                    std=(0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.ToTensor()
        
        ood_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        )
        return ood_dataset
    except Exception as e:
        print(f"Could not load CIFAR-100: {e}")
        return None


def calculate_fpr_at_tpr(y_true, scores, target_tpr=0.95):
    """
    Calculate FPR at specified TPR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Find threshold where TPR >= target
    valid_indices = np.where(tpr >= target_tpr)[0]
    if len(valid_indices) == 0:
        return 1.0  # Worst case
    
    # Return FPR at smallest threshold that achieves target TPR
    idx = valid_indices[0]
    return fpr[idx]


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
    ood_method = config.get("ood_method", "cifar100")  # default to CIFAR-100 like original
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"OOD method: {ood_method}")
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "ood",
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
            "evaluator": "ood",
            "success": False,
            "error": f"Failed to load model: {e}",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    # Load test data (in-distribution)
    test_data_path = input_dir / "test_data.npy"
    test_labels_path = input_dir / "test_labels.npy"
    
    if not test_data_path.exists() or not test_labels_path.exists():
        results = {
            "evaluator": "ood",
            "success": False,
            "error": "Test data not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    X_id = np.load(test_data_path)
    Y_id = np.load(test_labels_path)
    
    print(f"Loaded {len(X_id)} in-distribution samples")
    
    # Create in-distribution dataloader
    id_dataset = TensorDataset(
        torch.tensor(X_id).float(),
        torch.tensor(Y_id).long()
    )
    id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)
    
    # Get OOD data
    ood_loader = None
    n_ood_samples = 0
    
    if ood_method == "cifar100":
        # Try to use CIFAR-100 as OOD (like original implementation)
        ood_dataset = load_cifar100_ood(transform_normalize=True)
        if ood_dataset:
            ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)
            n_ood_samples = len(ood_dataset)
            print(f"Using CIFAR-100 as OOD: {n_ood_samples} samples")
        else:
            print("CIFAR-100 not available, falling back to gaussian")
            ood_method = "gaussian"
    
    if ood_loader is None:
        # Generate synthetic OOD samples
        X_ood = generate_ood_samples(X_id, method=ood_method)
        Y_ood = np.zeros(len(X_ood))
        n_ood_samples = len(X_ood)
        
        ood_dataset = TensorDataset(
            torch.tensor(X_ood).float(),
            torch.tensor(Y_ood).long()
        )
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)
        print(f"Generated {n_ood_samples} OOD samples using '{ood_method}' method")
    
    # Get confidence scores
    print("Computing scores for in-distribution samples...")
    id_scores = get_max_confidence_scores(model, id_loader, device)
    
    print("Computing scores for OOD samples...")
    ood_scores = get_max_confidence_scores(model, ood_loader, device)
    
    # Combine for evaluation
    # Labels: 1 = OOD (should have high score), 0 = in-distribution (should have low score)
    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    # Calculate metrics
    try:
        auc = roc_auc_score(all_labels, all_scores)
        fpr_at_95 = calculate_fpr_at_tpr(all_labels, all_scores, target_tpr=0.95)
        
        metrics = {
            "ood_auc": float(auc),
            "fpr_at_95_tpr": float(fpr_at_95)
        }
        
        print(f"OOD Detection AUC: {auc:.4f}")
        print(f"FPR at 95% TPR: {fpr_at_95:.4f}")
        
        results = {
            "evaluator": "ood",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {
                "ood_method": ood_method,
                "n_id_samples": len(X_id),
                "n_ood_samples": n_ood_samples,
                "score_formula": "1 - max_confidence"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        results = {
            "evaluator": "ood",
            "success": False,
            "error": str(e),
            "metrics": {}
        }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
