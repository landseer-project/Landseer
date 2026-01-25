#!/usr/bin/env python3
"""
Adversarial Robustness Evaluator

Evaluates model robustness against:
- PGD (Projected Gradient Descent) attack
- FGSM (Fast Gradient Sign Method) attack  
- Carlini L2 attack (optimized implementation from src_old)

This matches the original functionality from src_old/landseer_pipeline/evaluator/
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CarliniL2Attack:
    """
    Optimized Carlini & Wagner L2 Attack implementation
    Adapted from src_old/landseer_pipeline/evaluator/carlini_attack.py
    """
    def __init__(self, model, device, confidence=0, learning_rate=0.01, 
                 max_iterations=50, binary_search_steps=1, initial_const=1e-3,
                 early_stop_threshold=1e-4):
        self.model = model
        self.device = device
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.early_stop_threshold = early_stop_threshold
        
    def __call__(self, images, labels):
        """Generate adversarial examples using optimized C&W L2 attack"""
        batch_size = images.shape[0]
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        # Initialize variables
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        const = torch.full((batch_size,), self.initial_const, device=self.device)
        
        best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        best_adv = images.clone()
        
        # Pre-compute mask for efficiency
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # Binary search with early stopping
        for search_step in range(self.binary_search_steps):
            # Initialize perturbation
            delta = torch.zeros_like(images, requires_grad=True)
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            
            prev_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                
                adv_images = torch.clamp(images + delta, 0, 1)
                outputs = self.model(adv_images)
                
                # Vectorized loss calculation
                l2_loss = torch.norm((adv_images - images).view(batch_size, -1), 
                                   p=2, dim=1) ** 2
                
                # Efficient Carlini objective
                real_scores = outputs[batch_indices, labels]
                outputs_copy = outputs.clone()
                outputs_copy[batch_indices, labels] = -1e4
                other_scores = torch.max(outputs_copy, dim=1)[0]
                
                f_loss = torch.clamp(real_scores - other_scores + self.confidence, min=0)
                total_loss = torch.sum(const * f_loss + l2_loss)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Early stopping check
                if abs(prev_loss - total_loss.item()) < self.early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break
                else:
                    patience_counter = 0
                prev_loss = total_loss.item()
                
                # Update best examples every 10 iterations
                if iteration % 10 == 0:
                    with torch.no_grad():
                        pred_labels = torch.argmax(outputs, dim=1)
                        success_mask = (pred_labels != labels) & (l2_loss < best_l2)
                        best_l2[success_mask] = l2_loss[success_mask]
                        best_adv[success_mask] = adv_images[success_mask]
            
            # Vectorized binary search update
            with torch.no_grad():
                final_outputs = self.model(best_adv)
                final_pred = torch.argmax(final_outputs, dim=1)
                success_mask = (final_pred != labels)
                
                upper_bound[success_mask] = const[success_mask]
                lower_bound[~success_mask] = const[~success_mask]
                
                # Update const
                valid_upper = upper_bound < 1e9
                const[valid_upper] = (lower_bound[valid_upper] + upper_bound[valid_upper]) / 2
                const[~valid_upper] = lower_bound[~valid_upper] * 10
        
        return best_adv


def load_model(model_path: Path, device: str):
    """Load PyTorch model from path."""
    model = torch.load(model_path, map_location=device)
    if isinstance(model, dict):
        raise ValueError("Model is a state dict, need full model")
    model.eval()
    return model


def evaluate_clean_accuracy(model, loader, device):
    """Evaluate clean (no attack) accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_pgd(model, loader, device, eps=8/255, alpha=2/255, steps=10):
    """
    Evaluate robustness against PGD attack.
    Uses same parameters as src_old: eps=8/255, alpha=2/255, steps=10
    """
    try:
        from torchattacks import PGD
        attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
    except ImportError:
        print("torchattacks not available, using custom PGD")
        return evaluate_pgd_custom(model, loader, device, eps, alpha, steps)
    
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_pgd_custom(model, loader, device, eps=8/255, alpha=2/255, steps=10):
    """Custom PGD implementation as fallback."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Initialize perturbation
        delta = torch.zeros_like(images, requires_grad=True)
        
        for _ in range(steps):
            outputs = model(images + delta)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            # Update perturbation
            delta.data = delta.data + alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            delta.grad.zero_()
        
        with torch.no_grad():
            outputs = model(images + delta)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_fgsm(model, loader, device, eps=8/255):
    """Evaluate robustness against FGSM attack."""
    try:
        from torchattacks import FGSM
        attack = FGSM(model, eps=eps)
    except ImportError:
        print("torchattacks not available, using custom FGSM")
        return evaluate_fgsm_custom(model, loader, device, eps)
    
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        adv_images = attack(images, labels)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_fgsm_custom(model, loader, device, eps=8/255):
    """Custom FGSM implementation as fallback."""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # Create adversarial example
        adv_images = images + eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_carlini_l2(model, loader, device, confidence=0, sample_size=500):
    """
    Evaluate robustness against Carlini L2 attack.
    Uses optimized implementation from src_old/landseer_pipeline/evaluator/carlini_attack.py
    """
    model.eval()
    attack = CarliniL2Attack(
        model, device, confidence=confidence,
        max_iterations=50,
        binary_search_steps=1
    )
    
    # Sample random subset for faster evaluation
    
    all_X, all_y = [], []
    for X, y in loader:
        all_X.append(X)
        all_y.append(y)
        if len(all_X) * X.size(0) >= sample_size:
            break
    
    if not all_X:
        return 0.0
    
    # Combine and sample
    combined_X = torch.cat(all_X, dim=0)[:sample_size]
    combined_y = torch.cat(all_y, dim=0)[:sample_size]
    
    # Single batch evaluation
    combined_X = combined_X.to(device, non_blocking=True)
    combined_y = combined_y.to(device, non_blocking=True)
    
    adv_X = attack(combined_X, combined_y)
    
    with torch.no_grad():
        pred = model(adv_X).argmax(1)
        correct = (pred == combined_y).sum().item()
    
    return correct / len(combined_y)


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
    
    eps = config.get("eps", 8/255)
    pgd_steps = config.get("pgd_steps", 10)
    pgd_alpha = config.get("pgd_alpha", 2/255)
    batch_size = config.get("batch_size", 64)
    carlini_sample_size = config.get("carlini_sample_size", 500)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "adversarial",
            "success": False,
            "error": "model.pt not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    try:
        model = load_model(model_path, device)
    except Exception as e:
        results = {
            "evaluator": "adversarial",
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
            "evaluator": "adversarial",
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
    
    # Evaluate
    metrics = {}
    
    print("Evaluating clean accuracy...")
    metrics["clean_accuracy"] = evaluate_clean_accuracy(model, loader, device)
    print(f"  Clean accuracy: {metrics['clean_accuracy']:.4f}")
    
    print(f"Evaluating PGD robustness (eps={eps}, alpha={pgd_alpha}, steps={pgd_steps})...")
    metrics["pgd_accuracy"] = evaluate_pgd(model, loader, device, eps=eps, alpha=pgd_alpha, steps=pgd_steps)
    print(f"  PGD accuracy: {metrics['pgd_accuracy']:.4f}")
    
    print(f"Evaluating FGSM robustness (eps={eps})...")
    metrics["fgsm_accuracy"] = evaluate_fgsm(model, loader, device, eps=eps)
    print(f"  FGSM accuracy: {metrics['fgsm_accuracy']:.4f}")
    
    print(f"Evaluating Carlini L2 robustness (sample_size={carlini_sample_size})...")
    metrics["carlini_l2_accuracy"] = evaluate_carlini_l2(model, loader, device, sample_size=carlini_sample_size)
    print(f"  Carlini L2 accuracy: {metrics['carlini_l2_accuracy']:.4f}")
    
    # Write results
    results = {
        "evaluator": "adversarial",
        "success": True,
        "skipped": False,
        "metrics": metrics,
        "parameters": {
            "eps": float(eps),
            "pgd_steps": pgd_steps,
            "pgd_alpha": float(pgd_alpha),
            "carlini_sample_size": carlini_sample_size,
            "device": device
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nEvaluation complete. Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
