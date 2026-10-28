#!/usr/bin/env python3

"""
Watermark Verification Evaluator

Supports:
1. Weight-based Uchida watermarking -> bit_accuracy
2. Trigger-based watermarking -> wmacc_badnets
"""

import json
import os
from datetime import datetime
from pathlib import Path
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_loader import load_torch_model_for_eval


def find_uchida_style_conv2d(model):
    """Use the same layer selected by the Uchida defense."""
    try:
        conv_layer = model.layer1[0].conv2
        layer_name = "layer1.0.conv2"
        print(f"Selected Uchida watermark layer: {layer_name} -> {conv_layer}")
        return layer_name, conv_layer
    except Exception as e:
        print(f"Failed to locate Uchida watermark layer: {e}")
        return None, None


def decode_watermark(conv_layer, wm_matrix, wm_bits, device):
    """Decode Uchida-style watermark bits from the selected Conv2D layer."""
    try:
        with torch.no_grad():
            weight = conv_layer.weight
            mean_weight = weight.mean(dim=[2, 3])
            flat = mean_weight.view(1, -1)

            print(
                f"Watermark decode shapes: weight={tuple(weight.shape)}, "
                f"flat={tuple(flat.shape)}, wm_matrix={tuple(wm_matrix.shape)}, "
                f"wm_bits={tuple(wm_bits.shape)}"
            )

            if flat.shape[1] != wm_matrix.shape[0]:
                raise ValueError(
                    f"Watermark dimension mismatch: layer produces {flat.shape[1]} "
                    f"features, but wm_matrix expects {wm_matrix.shape[0]}"
                )

            projection = torch.sigmoid(flat @ wm_matrix)
            decoded_bits = (projection > 0.5).float()
            wm_bits = wm_bits.reshape_as(decoded_bits)
            accuracy = (decoded_bits == wm_bits).float().mean().item()

            print(f"Decoded {int(decoded_bits.sum().item())}/{decoded_bits.numel()} bits as 1")
            return accuracy, decoded_bits.cpu().numpy()

    except Exception as e:
        print(f"Error in watermark decoding: {e}")
        return 0.0, None


def evaluate_weight_based_watermark(model, wm_matrix_path, wm_bits_path, device):
    print("Evaluating weight-based watermarking...")
    print(f"Watermark matrix: {wm_matrix_path}")
    print(f"Watermark bits: {wm_bits_path}")

    wm_matrix = torch.tensor(np.load(wm_matrix_path), dtype=torch.float32).to(device)
    wm_bits = torch.tensor(np.load(wm_bits_path), dtype=torch.float32).to(device)

    layer_name, conv_layer = find_uchida_style_conv2d(model)
    if conv_layer is None:
        return 0.0, None

    accuracy, _ = decode_watermark(conv_layer, wm_matrix, wm_bits, device)
    print(f"Weight-based watermark decoding accuracy: {accuracy * 100:.2f}%")

    return accuracy, {
        "method": "weight_based",
        "layer_name": layer_name,
        "expected_bits": len(wm_bits.view(-1)),
        "wm_matrix": str(wm_matrix_path),
        "wm_bits": str(wm_bits_path),
    }


def resolve_watermark_artifacts(input_dir):
    """
    Resolve dataset-specific reference watermark artifacts.

    Landseer mounts the required 'watermark' artifact directory at:
        input/watermark/

    Expected layout:
        input/watermark/
            cifar10/
                wm_matrix.npy
                wm_bits.npy
                wm_images.npy
                wm_labels.npy
            celeba/
                wm_matrix.npy
                wm_bits.npy
                wm_images.npy
                wm_labels.npy
    """
    artifact_dir = input_dir / "watermark"

    dataset = os.environ.get("LANDSEER_DATASET", "").strip().lower()
    variant = os.environ.get("LANDSEER_DATASET_VARIANT", "").strip().lower()

    if dataset == "cifar10":
        dataset_dir = artifact_dir / "cifar10"
    elif dataset == "celeba":
        dataset_dir = artifact_dir / "celeba"
    else:
        raise ValueError(
            f"No watermark artifacts configured for dataset={dataset!r}, "
            f"variant={variant!r}"
        )

    return {
        "dataset": dataset,
        "variant": variant,
        "dataset_dir": dataset_dir,
        "wm_matrix": dataset_dir / "wm_matrix.npy",
        "wm_bits": dataset_dir / "wm_bits.npy",
        "triggers": dataset_dir,
    }


def resolve_watermark_sources(input_dir):
    artifacts = resolve_watermark_artifacts(input_dir)

    # Uchida priority:
    # 1. Upstream Uchida tool output
    # 2. Dataset-specific reference artifacts
    if (input_dir / "wm_matrix.npy").exists() and (input_dir / "wm_bits.npy").exists():
        wm_matrix_path = input_dir / "wm_matrix.npy"
        wm_bits_path = input_dir / "wm_bits.npy"
        print("Using upstream Uchida wm_matrix.npy / wm_bits.npy")
    else:
        wm_matrix_path = artifacts["wm_matrix"]
        wm_bits_path = artifacts["wm_bits"]
        print(f"Using reference Uchida weights: {wm_matrix_path.parent}")

    # Trigger priority:
    # 1. Upstream trigger watermark tool output
    # 2. Upstream watermark_triggers directory
    # 3. Dataset-specific reference artifacts
    if (input_dir / "wm_test_data.npy").exists() and (input_dir / "wm_test_labels.npy").exists():
        triggers_path = input_dir
        print("Using upstream BadNets wm_test_data.npy / wm_test_labels.npy")
    elif (input_dir / "watermark_triggers").exists():
        triggers_path = input_dir / "watermark_triggers"
        print("Using upstream watermark_triggers directory")
    else:
        triggers_path = artifacts["triggers"]
        print(f"Using reference trigger watermark: {triggers_path}")

    return artifacts["dataset"], wm_matrix_path, wm_bits_path, triggers_path


def evaluate_trigger_based_watermark(model, triggers_path, device, dataset_type):
    """Evaluate trigger-based watermark accuracy."""
    from torchvision import transforms

    print(f"Evaluating trigger-based watermarking from: {triggers_path}")
    print(f"LANDSEER_DATASET: {dataset_type}")

    trigger_images = None
    trigger_labels = None
    image_file = None
    label_file = None

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

    if trigger_labels is None:
        for labels_txt in ["labels.txt", "labels-cifar.txt"]:
            txt_path = triggers_path / labels_txt
            if txt_path.exists():
                trigger_labels = np.loadtxt(txt_path).astype(np.int64)
                label_file = labels_txt
                break

    if trigger_images is None:
        image_files = glob(str(triggers_path / "*.png")) + glob(str(triggers_path / "pics/*.png"))

        if image_files and trigger_labels is not None:
            try:
                from PIL import Image

                images = []
                for img_file in sorted(image_files)[:len(trigger_labels)]:
                    img = Image.open(img_file).convert("RGB")
                    images.append(transforms.ToTensor()(img).numpy())

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
        raise ValueError(
            f"Trigger image/label count mismatch: "
            f"{len(trigger_images)} vs {len(trigger_labels)}"
        )

    data_min = float(trigger_images.min())
    data_max = float(trigger_images.max())

    print(f"Trigger image file: {image_file}")
    print(f"Trigger label file: {label_file}")
    print(f"Loaded {len(trigger_images)} trigger images")
    print(f"Trigger shape: {trigger_images.shape}")
    print(f"Trigger range: {data_min:.3f} to {data_max:.3f}")

    if data_min < -1e-6 or data_max > 1.0001:
        raise ValueError(
            f"Expected trigger images in [0,1], "
            f"got range [{data_min:.3f},{data_max:.3f}]"
        )

    # Dataset comes from Landseer, not image dimensions.
    if dataset_type == "celeba":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif dataset_type == "cifar10":
        preprocess = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
    else:
        raise ValueError(f"Unsupported watermark dataset: {dataset_type!r}")

    dataset = TensorDataset(
        torch.tensor(trigger_images).float(),
        torch.tensor(trigger_labels).long()
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    prediction_counts = {}

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
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
        "n_triggers": total,
    }


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = input_dir / "model.pt"
    watermark_key_path = input_dir / "watermark_key.json"

    if not model_path.exists():
        results = {
            "evaluator": "watermark",
            "success": False,
            "error": "model.pt not found",
            "metrics": {},
        }
        (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
        return

    try:
        dataset_type, wm_matrix_path, wm_bits_path, triggers_path = resolve_watermark_sources(input_dir)

        print(f"Device: {device}")
        print(f"LANDSEER_DATASET: {os.environ.get('LANDSEER_DATASET')}")
        print(f"LANDSEER_DATASET_VARIANT: {os.environ.get('LANDSEER_DATASET_VARIANT')}")
        print(f"Resolved wm_matrix: {wm_matrix_path}")
        print(f"Resolved wm_bits: {wm_bits_path}")
        print(f"Resolved triggers: {triggers_path}")

        has_weight_based = wm_matrix_path.exists() and wm_bits_path.exists()

        has_trigger_based = any([
            (triggers_path / "wm_test_data.npy").exists() and (triggers_path / "wm_test_labels.npy").exists(),
            (triggers_path / "wm_images.npy").exists() and (triggers_path / "wm_labels.npy").exists(),
            (triggers_path / "trigger_images.npy").exists() and (triggers_path / "trigger_labels.npy").exists(),
            (triggers_path / "watermark_images.npy").exists() and (triggers_path / "watermark_labels.npy").exists(),
        ])

        has_watermark_key = watermark_key_path.exists()

        if not (has_weight_based or has_trigger_based or has_watermark_key):
            results = {
                "evaluator": "watermark",
                "success": True,
                "skipped": True,
                "skip_reason": "No watermark artifacts found",
                "metrics": {
                    "watermark_accuracy": None,
                    "bit_accuracy": None,
                    "detection_rate": None,
                    "wmacc_badnets": None,
                },
            }
            (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
            print("Skipping: No watermark data found")
            return

        model = load_torch_model_for_eval(model_path, input_dir, device)

        weight_based_accuracy = None
        trigger_based_accuracy = None
        weight_info = None
        trigger_info = None

        if has_weight_based:
            weight_based_accuracy, weight_info = evaluate_weight_based_watermark(
                model, str(wm_matrix_path), str(wm_bits_path), device
            )

        if has_trigger_based:
            trigger_based_accuracy, trigger_info = evaluate_trigger_based_watermark(
                model, triggers_path, device, dataset_type
            )

        # Legacy watermark_key.json fallback only if no Uchida key exists.
        if has_watermark_key and weight_based_accuracy is None:
            try:
                watermark_key = json.loads(watermark_key_path.read_text())
                expected_watermark = watermark_key.get("watermark", [])

                if expected_watermark:
                    layer_name, conv_layer = find_uchida_style_conv2d(model)

                    if conv_layer:
                        weights = conv_layer.weight.data.cpu().numpy().flatten()
                        watermark_length = len(expected_watermark)

                        if len(weights) >= watermark_length:
                            extracted = (weights[:watermark_length] > 0).astype(int)
                            correct_bits = np.sum(extracted == np.array(expected_watermark))
                            weight_based_accuracy = correct_bits / watermark_length
                            weight_info = {
                                "method": "weight_sign",
                                "watermark_length": watermark_length,
                                "layer_name": layer_name,
                            }
            except Exception as e:
                print(f"Failed to parse watermark_key.json: {e}")

        # Legacy/general watermark score.
        if weight_based_accuracy is not None:
            watermark_accuracy = weight_based_accuracy
        elif trigger_based_accuracy is not None:
            watermark_accuracy = trigger_based_accuracy
        else:
            watermark_accuracy = 0.0

        detection_threshold = 0.7
        detected = watermark_accuracy >= detection_threshold
        detection_rate = 1.0 if detected else 0.0

        metrics = {
            "watermark_accuracy": float(watermark_accuracy),
            "bit_accuracy": float(weight_based_accuracy) if weight_based_accuracy is not None else None,
            "detection_rate": float(detection_rate),
            "wmacc_badnets": float(trigger_based_accuracy) if trigger_based_accuracy is not None else None,
        }

        print(f"\nUchida bit accuracy: {weight_based_accuracy}")
        print(f"Trigger watermark accuracy: {trigger_based_accuracy}")
        print(f"Final watermark accuracy: {watermark_accuracy:.4f}")
        print(f"Detection: {'YES' if detected else 'NO'}")

        results = {
            "evaluator": "watermark",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {
                "dataset": dataset_type,
                "detection_threshold": detection_threshold,
                "weight_based": weight_info,
                "trigger_based": trigger_info,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        results = {
            "evaluator": "watermark",
            "success": False,
            "error": str(e),
            "metrics": {},
        }
        print(f"Watermark evaluation failed: {e}")

    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()