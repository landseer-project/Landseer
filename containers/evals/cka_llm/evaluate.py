#!/usr/bin/env python3

"""
REEF CKA Evaluator

Computes linear CKA similarity between two sets of saved model
activations and reports the mean corresponding-layer CKA as `cka_sim`.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# ============================================================
# CKA
# ============================================================

class CudaCKA:
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        H = torch.eye(n, device=self.device, dtype=K.dtype) - torch.ones((n, n), device=self.device, dtype=K.dtype) / n
        return H @ K @ H

    def linear_HSIC(self, X, Y):
        return torch.sum(self.centering(X @ X.T) * self.centering(Y @ Y.T))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return hsic / (var1 * var2)


# ============================================================
# Activation loading
# ============================================================

def find_layers(model_dir):
    """Return available layer numbers from layer_<layer>_<batch>.pt files."""
    layers = set()

    for path in model_dir.glob("layer_*_*.pt"):
        parts = path.stem.split("_")
        if len(parts) >= 3:
            layers.add(int(parts[1]))

    return sorted(layers)


def load_layer_acts(model_dir, layer, device, center=True, scale=True):
    """Load and preprocess all saved activation chunks for one layer."""
    files = sorted(
        model_dir.glob(f"layer_{layer}_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    if not files:
        raise FileNotFoundError(f"No activation files found for layer {layer} in {model_dir}")

    acts = torch.cat([torch.load(f, map_location="cpu") for f in files], dim=0).float().to(device)

    if center:
        acts = acts - torch.mean(acts, dim=0)

    if scale:
        std = torch.std(acts, dim=0)
        acts = acts / std.clamp_min(1e-8)

    return acts


# ============================================================
# Metadata
# ============================================================

def load_metadata(model_dir):
    metadata_path = model_dir / "metadata.json"
    return json.loads(metadata_path.read_text()) if metadata_path.exists() else None


def validate_metadata(base_metadata, test_metadata):
    """Ensure both activation sets were generated from identical inputs."""
    if base_metadata is None or test_metadata is None:
        return

    for key in ["sample_count", "token_position", "dataset_sha256"]:
        if base_metadata.get(key) != test_metadata.get(key):
            raise ValueError(
                f"Activation metadata mismatch for '{key}': "
                f"{base_metadata.get(key)} != {test_metadata.get(key)}"
            )


# ============================================================
# CKA evaluation
# ============================================================

def compute_cka_similarity(base_dir, test_dir, device):
    base_layers = find_layers(base_dir)
    test_layers = find_layers(test_dir)

    if not base_layers:
        raise ValueError(f"No activation layers found in {base_dir}")

    if not test_layers:
        raise ValueError(f"No activation layers found in {test_dir}")

    common_layers = sorted(set(base_layers) & set(test_layers))

    if not common_layers:
        raise ValueError("The two models have no common layer indices.")

    print(f"Base layers: {base_layers}")
    print(f"Test layers: {test_layers}")
    print(f"Corresponding layers: {common_layers}")

    cka = CudaCKA(device)
    diagonal_scores = []

    for layer in common_layers:
        X = load_layer_acts(base_dir, layer, device, center=True, scale=True)
        Y = load_layer_acts(test_dir, layer, device, center=True, scale=True)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Sample count mismatch at layer {layer}: {X.shape[0]} vs {Y.shape[0]}")

        score = float(cka.linear_CKA(X, Y).detach().cpu())
        diagonal_scores.append(score)

        print(f"Layer {layer}: CKA = {score:.6f}")

    cka_sim = float(np.mean(diagonal_scores))

    return cka_sim, common_layers, diagonal_scores


# ============================================================
# Output helper
# ============================================================

def save_results(output_dir, results):
    result_path = output_dir / "evaluation_results.json"
    result_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {result_path}")


# ============================================================
# Main
# ============================================================

def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    config_path = input_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    base_model = config.get("base_model", "gemma-2-2b")
    test_model = config.get("test_model", "gemma-2-2b-it")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = input_dir / base_model
    test_dir = input_dir / test_model

    print("=" * 60)
    print("REEF CKA EVALUATION")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Base model:      {base_model}")
    print(f"Test model:      {test_model}")
    print(f"Device:          {device}")
    print("=" * 60)

    if not base_dir.exists():
        save_results(output_dir, {
            "evaluator": "reef_cka",
            "success": False,
            "error": f"Base activation directory not found: {base_dir}",
            "metrics": {},
        })
        return

    if not test_dir.exists():
        save_results(output_dir, {
            "evaluator": "reef_cka",
            "success": False,
            "error": f"Test activation directory not found: {test_dir}",
            "metrics": {},
        })
        return

    try:
        base_metadata = load_metadata(base_dir)
        test_metadata = load_metadata(test_dir)
        validate_metadata(base_metadata, test_metadata)

        cka_sim, layers, layer_scores = compute_cka_similarity(base_dir, test_dir, device)

        print(f"\nMean diagonal CKA: {cka_sim:.6f}")

        results = {
            "evaluator": "reef_cka",
            "success": True,
            "skipped": False,
            "metrics": {
                "cka_sim": cka_sim,
            },
            "details": {
                "base_model": base_model,
                "test_model": test_model,
                "layers": layers,
                "layer_cka": layer_scores,
            },
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    except Exception as e:
        results = {
            "evaluator": "reef_cka",
            "success": False,
            "error": str(e),
            "metrics": {},
        }

    save_results(output_dir, results)


if __name__ == "__main__":
    main()
