#!/usr/bin/env python3
"""
Watermark Verification Evaluator

Evaluates watermark detection and extraction accuracy.
Supports both:
1. Weight-based watermarking (Uchida-style) - extracts watermark from Conv2D weights
2. Trigger-based watermarking (WatermarkNN) - tests on trigger images

This matches the original functionality from src_old/landseer_pipeline/evaluator/model_evaluator.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_loader import load_torch_model_for_eval


def find_uchida_style_conv2d(model):
    """
    Find appropriate Conv2D layer for watermark embedding.
    Adapted from src_old/landseer_pipeline/evaluator/model_evaluator.py
    
    Prefers second-last Conv2D layer, falls back to last.
    """
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    if not conv_layers:
        return None, None
    
    # Prefer second-last conv if exists, else fallback to last
    if len(conv_layers) >= 2:
        selected_idx = -2
    else:
        selected_idx = -1
    
    selected_name, selected_layer = conv_layers[selected_idx]
    print(f"Selected Conv2D layer for watermark evaluation: {selected_name}")
    return selected_name, selected_layer


def decode_watermark(conv_layer, wm_matrix, wm_bits, device):
    """
    Decode watermark from Conv2D layer weights.
    Adapted from src_old/landseer_pipeline/evaluator/model_evaluator.py
    
    Uses Uchida-style watermarking (weight-based):
    1. Extract mean weights from conv filters
    2. Project through watermark matrix
    3. Apply sigmoid and threshold at 0.5
    """
    try:
        with torch.no_grad():
            weight = conv_layer.weight
            mean_weight = weight.mean(dim=[2, 3])  # Average over spatial dimensions
            flat = mean_weight.view(1, -1)
            
            # Handle dimension mismatch
            expected_dim = wm_matrix.shape[0]
            if flat.shape[1] != expected_dim:
                if flat.shape[1] > expected_dim:
                    flat = flat[:, :expected_dim]
                else:
                    padding_size = expected_dim - flat.shape[1]
                    flat = torch.cat([flat, torch.zeros(1, padding_size, device=device)], dim=1)
            
            projection = torch.sigmoid(flat @ wm_matrix)
            decoded_bits = (projection > 0.5).float()
            accuracy = (decoded_bits == wm_bits).float().mean().item()
            
            return accuracy, decoded_bits.cpu().numpy()
    except Exception as e:
        print(f"Error in watermark decoding: {e}")
        return 0.0, None


def evaluate_weight_based_watermark(model, wm_matrix_path, wm_bits_path, device):
    """
    Evaluate weight-based watermarking (Uchida-style).
    """
    print("Evaluating weight-based watermarking...")
    
    # Load watermark matrix and bits
    wm_matrix = torch.tensor(np.load(wm_matrix_path), dtype=torch.float32).to(device)
    wm_bits = torch.tensor(np.load(wm_bits_path), dtype=torch.float32).to(device)
    
    # Find appropriate Conv2D layer
    layer_name, conv_layer = find_uchida_style_conv2d(model)
    if conv_layer is None:
        print("No Conv2D layer found for watermark evaluation")
        return 0.0, None
    
    # Decode watermark
    accuracy, decoded_bits = decode_watermark(conv_layer, wm_matrix, wm_bits, device)
    print(f"Weight-based watermark decoding accuracy: {accuracy * 100:.2f}%")
    
    return accuracy, {
        "method": "weight_based",
        "layer_name": layer_name,
        "expected_bits": len(wm_bits.view(-1))
    }


# def evaluate_trigger_based_watermark(model, triggers_path, device):
#     """
#     Evaluate trigger-based watermarking (WatermarkNN style).
#     Tests model accuracy on watermark trigger images.
#     """
#     print(f"Evaluating trigger-based watermarking from: {triggers_path}")
    
#     # Load triggers
#     trigger_images = None
#     trigger_labels = None
    
#     # Check for numpy format
#     for img_name in ["trigger_images.npy", "watermark_images.npy", "wm_images.npy"]:
#         img_path = triggers_path / img_name
#         if img_path.exists():
#             trigger_images = np.load(img_path)
#             break
    
#     for label_name in ["trigger_labels.npy", "watermark_labels.npy", "wm_labels.npy", "labels.npy"]:
#         label_path = triggers_path / label_name
#         if label_path.exists():
#             trigger_labels = np.load(label_path)
#             break
    
#     # Try text labels file (WatermarkNN format)
#     if trigger_labels is None:
#         for labels_txt in ["labels.txt", "labels-cifar.txt"]:
#             txt_path = triggers_path / labels_txt
#             if txt_path.exists():
#                 trigger_labels = np.loadtxt(txt_path)
#                 break
    
#     # Try loading from image folder if numpy not found
#     if trigger_images is None:
#         image_files = glob(str(triggers_path / "*.png")) + glob(str(triggers_path / "pics/*.png"))
#         if image_files and trigger_labels is not None:
#             try:
#                 from PIL import Image
#                 from torchvision import transforms
                
#                 transform = transforms.Compose([
#                     transforms.Resize(32),
#                     transforms.CenterCrop(32),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ])
                
#                 images = []
#                 for img_file in sorted(image_files)[:len(trigger_labels)]:
#                     img = Image.open(img_file).convert('RGB')
#                     img_tensor = transform(img)
#                     images.append(img_tensor.numpy())
                
#                 trigger_images = np.stack(images)
#             except Exception as e:
#                 print(f"Failed to load images from folder: {e}")
    
#     if trigger_images is None or trigger_labels is None:
#         print("No trigger images or labels found")
#         return None, None
    
#     print(f"Loaded {len(trigger_images)} trigger images")
    
#     # Create dataloader
#     dataset = TensorDataset(
#         torch.tensor(trigger_images).float(),
#         torch.tensor(trigger_labels).long()
#     )
#     loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
#     # Evaluate
#     model.eval()
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for data, targets in loader:
#             data, targets = data.to(device), targets.to(device)
#             outputs = model(data)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
    
#     accuracy = correct / total if total > 0 else 0.0
#     print(f"Trigger-based watermark accuracy: {accuracy * 100:.2f}%")
    
#     return accuracy, {
#         "method": "trigger_based",
#         "n_triggers": total
#     }

def evaluate_trigger_based_watermark(model, triggers_path, device):
    """
    Evaluate BadNets-style trigger watermark accuracy.

    Supported numpy formats:
      CIFAR-10: (N,3,32,32), [0,1]
      CelebA:   (N,3,64,64), [0,1]
    """
    from torchvision import transforms

    print(f"Evaluating trigger-based watermarking from: {triggers_path}")

    trigger_images = None
    trigger_labels = None
    image_file = None
    label_file = None

    # Numpy trigger files
    for img_name in ["wm_test_data.npy", "trigger_images.npy", "watermark_images.npy", "wm_images.npy"]:
        img_path = triggers_path / img_name
        if img_path.exists():
            trigger_images = np.load(img_path).astype(np.float32)
            image_file = img_name
            break

    for label_name in ["wm_test_labels.npy", "trigger_labels.npy", "watermark_labels.npy", "wm_labels.npy"]:
        label_path = triggers_path / label_name
        if label_path.exists():
            trigger_labels = np.load(label_path).astype(np.int64)
            label_file = label_name
            break

    # WatermarkNN text labels
    if trigger_labels is None:
        for labels_txt in ["labels.txt", "labels-cifar.txt"]:
            txt_path = triggers_path / labels_txt
            if txt_path.exists():
                trigger_labels = np.loadtxt(txt_path).astype(np.int64)
                label_file = labels_txt
                break

    # WatermarkNN image-folder format
    if trigger_images is None:
        image_files = glob(str(triggers_path / "*.png")) + glob(str(triggers_path / "pics/*.png"))

        if image_files and trigger_labels is not None:
            try:
                from PIL import Image

                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ])

                images = []
                for img_file in sorted(image_files)[:len(trigger_labels)]:
                    img = Image.open(img_file).convert("RGB")
                    images.append(transform(img).numpy())

                trigger_images = np.stack(images).astype(np.float32)
                image_file = "PNG trigger images"

            except Exception as e:
                print(f"Failed to load images from folder: {e}")

    if trigger_images is None or trigger_labels is None:
        print("No trigger images or labels found")
        return None, None

    if trigger_images.ndim != 4 or trigger_images.shape[1] != 3:
        raise ValueError(f"Expected trigger images (N,3,H,W), got {trigger_images.shape}")

    if len(trigger_images) != len(trigger_labels):
        raise ValueError(f"Trigger image/label count mismatch: {len(trigger_images)} vs {len(trigger_labels)}")

    H, W = trigger_images.shape[2:]
    data_min = float(trigger_images.min())
    data_max = float(trigger_images.max())

    print(f"Trigger image file: {image_file}")
    print(f"Trigger label file: {label_file}")
    print(f"Loaded {len(trigger_images)} trigger images")
    print(f"Trigger shape: {trigger_images.shape}")
    print(f"Trigger range: {data_min:.3f} to {data_max:.3f}")

    if data_min < -1e-6 or data_max > 1.0001:
        raise ValueError(
            f"Expected trigger images in [0,1], got range [{data_min:.3f},{data_max:.3f}]"
        )

    # Detect dataset from current Landseer data format and match test preprocessing.
    if (H, W) == (64, 64):
        dataset_type = "celeba"
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif (H, W) == (32, 32):
        dataset_type = "cifar10"
        preprocess = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    else:
        raise ValueError(
            f"Unsupported trigger format: shape={(H, W)}, range=[{data_min:.3f},{data_max:.3f}]. "
            "Expected CIFAR-10 (32x32, [0,1]) or CelebA (64x64, [0,1])."
        )

    print(f"Detected trigger dataset: {dataset_type}")

    dataset = TensorDataset(torch.tensor(trigger_images).float(), torch.tensor(trigger_labels).long())
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    prediction_counts = {}

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)

            # Match deterministic test preprocessing used by the trainer.
            data = preprocess(data)

            outputs = model(data)
            predicted = outputs.argmax(1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            for label, count in zip(*torch.unique(predicted, return_counts=True)):
                label = int(label.item())
                prediction_counts[label] = prediction_counts.get(label, 0) + int(count.item())

    accuracy = correct / total if total > 0 else 0.0

    print(f"Prediction counts: {prediction_counts}")
    print(f"Trigger-based watermark accuracy: {accuracy * 100:.2f}%")

    return accuracy, {
        "method": "trigger_based",
        "dataset": dataset_type,
        "trigger_source": str(triggers_path),
        "trigger_images": image_file,
        "trigger_labels": label_file,
        "n_triggers": total
    }


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config_path = input_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check for watermark key (required file for graceful skip)
    watermark_key_path = input_dir / "watermark_key.json"

    # Also check for alternative watermark files
    wm_matrix_path = input_dir / "wm_matrix.npy"
    wm_bits_path = input_dir / "wm_bits.npy"

    # Trigger source priority:
    # 1. wm_test_* generated by an upstream BadNets pre-watermarking tool
    # 2. watermark_triggers directory supplied upstream
    # 3. watermark test set baked into this evaluator image
    if (input_dir / "wm_test_data.npy").exists() and (input_dir / "wm_test_labels.npy").exists():
        triggers_path = input_dir
        print("Using upstream BadNets wm_test_data.npy / wm_test_labels.npy")

    elif (input_dir / "watermark_triggers").exists():
        triggers_path = input_dir / "watermark_triggers"
        print("Using upstream watermark_triggers directory")

    else:
        triggers_path = Path("/app/watermark_triggers")
        print("Using baked-in watermark trigger set")
    
    # # Also check for alternative watermark files
    # wm_matrix_path = input_dir / "wm_matrix.npy"
    # wm_bits_path = input_dir / "wm_bits.npy"
    # triggers_path = input_dir / "watermark_triggers"

    # # Otherwise use the watermark test set baked into the evaluator image.
    # if not triggers_path.exists():
    #     triggers_path = Path("/app/watermark_triggers")
    
    has_weight_based = wm_matrix_path.exists() and wm_bits_path.exists()
    has_trigger_based = triggers_path.exists() and triggers_path.is_dir()
    has_watermark_key = watermark_key_path.exists()
    
    if not (has_weight_based or has_trigger_based or has_watermark_key):
        # Graceful skip - no watermarking data found
        results = {
            "evaluator": "watermark",
            "success": True,
            "skipped": True,
            "skip_reason": "No watermark data found (watermark_key.json, wm_matrix.npy/wm_bits.npy, or watermark_triggers/)",
            "metrics": {
                "watermark_accuracy": None,
                "bit_accuracy": None,
                "detection_rate": None,
                "wmacc_badnets": None,
            }
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        print("Skipping: No watermark data found")
        return
    
    # Load model
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        results = {
            "evaluator": "watermark",
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
            "evaluator": "watermark",
            "success": False,
            "error": f"Failed to load model: {e}",
            "metrics": {}
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return
    
    # Try weight-based watermarking first
    watermark_accuracy = None
    method_info = None
    
    if has_weight_based:
        watermark_accuracy, method_info = evaluate_weight_based_watermark(
            model, str(wm_matrix_path), str(wm_bits_path), device
        )
    
    # Try trigger-based if weight-based not found or failed
    if watermark_accuracy is None or watermark_accuracy == 0:
        if has_trigger_based:
            trigger_acc, trigger_info = evaluate_trigger_based_watermark(
                model, triggers_path, device
            )
            if trigger_acc is not None:
                watermark_accuracy = trigger_acc
                method_info = trigger_info
    
    # Parse watermark_key.json if available
    if has_watermark_key and watermark_accuracy is None:
        try:
            watermark_key = json.loads(watermark_key_path.read_text())
            expected_watermark = watermark_key.get("watermark", [])
            
            if expected_watermark:
                # Simple weight sign-based extraction
                layer_name, conv_layer = find_uchida_style_conv2d(model)
                if conv_layer:
                    weights = conv_layer.weight.data.cpu().numpy().flatten()
                    watermark_length = len(expected_watermark)
                    
                    if len(weights) >= watermark_length:
                        extracted = (weights[:watermark_length] > 0).astype(int)
                        correct_bits = np.sum(extracted == np.array(expected_watermark))
                        watermark_accuracy = correct_bits / watermark_length
                        method_info = {
                            "method": "weight_sign",
                            "watermark_length": watermark_length,
                            "layer_name": layer_name
                        }
                        print(f"Weight-sign watermark accuracy: {watermark_accuracy * 100:.2f}%")
        except Exception as e:
            print(f"Failed to parse watermark_key.json: {e}")
    
    if watermark_accuracy is None:
        watermark_accuracy = 0.0
        method_info = {"method": "unknown"}
    
    # Calculate detection threshold
    detection_threshold = 0.7
    detected = watermark_accuracy >= detection_threshold
    detection_rate = 1.0 if detected else 0.0
    
    metrics = {
        "watermark_accuracy": float(watermark_accuracy),
        "bit_accuracy": float(watermark_accuracy),  # Same for now
        "detection_rate": float(detection_rate)
    }
    if method_info and method_info.get("method") == "trigger_based":
        metrics["wmacc_badnets"] = float(watermark_accuracy)

    print(f"\nFinal watermark accuracy: {watermark_accuracy:.4f}")
    print(f"Detection: {'YES' if detected else 'NO'}")
    
    results = {
        "evaluator": "watermark",
        "success": True,
        "skipped": False,
        "metrics": metrics,
        "parameters": {
            "detection_threshold": detection_threshold,
            **(method_info or {})
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
