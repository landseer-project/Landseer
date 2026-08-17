#!/usr/bin/env python3
"""
Backdoor Evaluator

Evaluates backdoor attack success rate by:
1. Applying trigger to clean test samples
2. Measuring how many are misclassified to target class
3. Also measuring clean accuracy
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_loader import load_torch_model_for_eval


def add_trigger(images, trigger_info):
    """
    Apply trigger pattern to images based on attack info.
    """
    technique = trigger_info.get("technique", "badnets")
    
    # if technique == "badnets":
    #     trigger_size = trigger_info.get("trigger_size", 3)
    #     trigger_value = trigger_info.get("trigger_value", 1.0)
    #     trigger_position = trigger_info.get("trigger_position", "bottom_right")
        
    #     triggered = images.copy()
    #     _, c, h, w = triggered.shape
        
    #     if trigger_position == "bottom_right":
    #         triggered[:, :, h-trigger_size:, w-trigger_size:] = trigger_value
    #     elif trigger_position == "top_left":
    #         triggered[:, :, :trigger_size, :trigger_size] = trigger_value
    #     elif trigger_position == "bottom_left":
    #         triggered[:, :, h-trigger_size:, :trigger_size] = trigger_value
    #     elif trigger_position == "top_right":
    #         triggered[:, :, :trigger_size, w-trigger_size:] = trigger_value
    #     else:
    #         triggered[:, :, h-trigger_size:, w-trigger_size:] = trigger_value
        
    #     return triggered
    if technique == "badnets":
        trigger_size = int(trigger_info.get("trigger_size", 3))

        # if "trigger_value" not in trigger_info:
        #     raise ValueError(
        #         "BadNets evaluation requires trigger_value in poisoning metadata."
        #     )

        # trigger_value = float(trigger_info["trigger_value"])
        trigger_value = resolve_eval_trigger_value(images) 
        trigger_position = trigger_info.get(
            "trigger_position",
            "bottom_right"
        )

        triggered = images.copy()

        if triggered.ndim != 4:
            raise ValueError(
                f"Expected [N,C,H,W] test images, got {triggered.shape}"
            )

        _, _, h, w = triggered.shape

        if trigger_size <= 0 or trigger_size > min(h, w):
            raise ValueError(
                f"Invalid trigger_size={trigger_size} "
                f"for images of size {h}x{w}"
            )

        if trigger_position == "bottom_right":
            row_start = h - trigger_size
            col_start = w - trigger_size

        elif trigger_position == "bottom_left":
            row_start = h - trigger_size
            col_start = 0

        elif trigger_position == "top_right":
            row_start = 0
            col_start = w - trigger_size

        elif trigger_position == "top_left":
            row_start = 0
            col_start = 0

        elif trigger_position == "center":
            row_start = (h - trigger_size) // 2
            col_start = (w - trigger_size) // 2

        else:
            raise ValueError(
                f"Unsupported trigger_position: {trigger_position}"
            )

        triggered[
            :,
            :,
            row_start:row_start + trigger_size,
            col_start:col_start + trigger_size,
        ] = trigger_value

        return triggered
    
    elif technique == "blend":
        alpha = trigger_info.get("alpha", 0.2)
        # Generate random pattern (should match training)
        np.random.seed(42)  # For reproducibility
        _, c, h, w = images.shape
        pattern = np.random.rand(1, c, h, w).astype(np.float32)
        triggered = (1 - alpha) * images + alpha * pattern
        return np.clip(triggered, 0, 1)
    
    else:
        # Default: apply badnets-style trigger
        trigger_size = 3
        triggered = images.copy()
        _, c, h, w = triggered.shape
        triggered[:, :, h-trigger_size:, w-trigger_size:] = 1.0
        return triggered

def get_predictions(outputs):
    """
    Convert model outputs to integer class predictions.

    Supports:
        multiclass / binary two-logit: [N, K]
        binary single-logit:           [N] or [N, 1]
    """

    if outputs.ndim == 1:
        return (outputs >= 0).long()

    if outputs.ndim == 2 and outputs.shape[1] == 1:
        return (outputs[:, 0] >= 0).long()

    if outputs.ndim == 2 and outputs.shape[1] >= 2:
        return outputs.argmax(dim=1)

    raise ValueError(
        f"Unsupported model output shape: {tuple(outputs.shape)}"
    )

def resolve_eval_trigger_value(images):
    data_min = float(images.min())
    data_max = float(images.max())

    # [0, 1]
    if data_min >= 0.0 and data_max <= 1.0 + 1e-6:
        return 1.0

    # [-1, 1]
    if data_min >= -1.0 - 1e-6 and data_max <= 1.0 + 1e-6:
        return 1.0

    # [0, 255]
    if data_min >= 0.0 and data_max <= 255.0 + 1e-6:
        return 255.0

    # Generic fallback: use observed maximum
    return data_max

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
    
    # Check for required poisoning metadata
    poison_meta_path = input_dir / "poisoning_metadata.json"
    if not poison_meta_path.exists():
        # Graceful skip
        results = {
            "evaluator": "backdoor",
            "success": True,
            "skipped": True,
            "skip_reason": "poisoning_metadata.json not found",
            "metrics": {
                "attack_success_rate": None,
                "clean_accuracy_post_attack": None,
                "backdoor_robustness": None
            }
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        print("Skipping: poisoning_metadata.json not found")
        return
    
    # Load poisoning metadata
    poison_meta = json.loads(poison_meta_path.read_text())
    target_class = poison_meta.get("target_class", 0)
    # trigger_info = poison_meta.get("attack_info", {})
    trigger_info = poison_meta.get(
        "attack_info",
        poison_meta.get("trigger_info", {})
    )
    
    print(f"Target class: {target_class}")
    print(f"Attack technique: {trigger_info.get('technique', 'unknown')}")
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "backdoor",
            "success": False,
            "error": "model.pt not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
    except Exception as e:
        results = {
            "evaluator": "backdoor",
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
            "evaluator": "backdoor",
            "success": False,
            "error": "Test data not found",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    X = np.load(test_data_path)
    Y = np.load(test_labels_path)
    
    print(f"Loaded {len(X)} test samples")

    print(
        f"Final test data: shape={X.shape}, "
        f"dtype={X.dtype}, "
        f"range=[{X.min():.4f}, {X.max():.4f}]"
    )

    print(
        f"Resolved BadNets trigger value: "
        f"{resolve_eval_trigger_value(X)}"
    )

    
    
    # Filter to non-target class samples for ASR calculation
    non_target_mask = Y != target_class
    X_non_target = X[non_target_mask]
    Y_non_target = Y[non_target_mask]
    
    print(f"Using {len(X_non_target)} non-target class samples for ASR")
    
    # Evaluate clean accuracy
    dataset = TensorDataset(
        torch.tensor(X).float(),
        torch.tensor(Y).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct_clean = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # _, predicted = outputs.max(1)
            predicted = get_predictions(outputs)
            total += labels.size(0)
            correct_clean += predicted.eq(labels).sum().item()
    
    clean_accuracy = correct_clean / total if total > 0 else 0.0
    
    # Evaluate attack success rate
    X_triggered = add_trigger(X_non_target, trigger_info)
    
    triggered_dataset = TensorDataset(
        torch.tensor(X_triggered).float(),
        torch.tensor(Y_non_target).long()
    )
    triggered_loader = DataLoader(triggered_dataset, batch_size=batch_size, shuffle=False)
    
    attack_success = 0
    total_triggered = 0
    
    with torch.no_grad():
        for images, _ in triggered_loader:
            images = images.to(device)
            outputs = model(images)
            # _, predicted = outputs.max(1)
            predicted = get_predictions(outputs)
            
            # Attack succeeds if predicted == target_class
            attack_success += (predicted == target_class).sum().item()
            total_triggered += images.size(0)
    
    asr = attack_success / total_triggered if total_triggered > 0 else 0.0
    
    # Backdoor robustness = 1 - ASR (higher is better, more resistant to backdoor)
    backdoor_robustness = 1.0 - asr
    
    metrics = {
        "clean_accuracy_post_attack": clean_accuracy,
        "attack_success_rate": asr,
        "backdoor_robustness": backdoor_robustness
    }
    
    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Attack success rate: {asr:.4f}")
    print(f"Backdoor robustness: {backdoor_robustness:.4f}")
    
    results = {
        "evaluator": "backdoor",
        "success": True,
        "skipped": False,
        "metrics": metrics,
        "parameters": {
            "target_class": target_class,
            "technique": trigger_info.get("technique", "unknown")
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
