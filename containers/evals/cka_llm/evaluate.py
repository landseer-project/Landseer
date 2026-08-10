#!/usr/bin/env python3

"""
REEF CKA Evaluator

Computes linear CKA similarity between two sets of saved
model activations and reports the mean corresponding-layer
CKA as the Landseer metric `cka_sim`.
"""

import json
import os
from datetime import datetime
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

        unit = torch.ones(
            (n, n),
            device=self.device,
            dtype=K.dtype,
        )

        identity = torch.eye(
            n,
            device=self.device,
            dtype=K.dtype,
        )

        H = identity - unit / n

        return H @ K @ H

    def linear_HSIC(self, X, Y):

        L_X = X @ X.T
        L_Y = Y @ Y.T

        return torch.sum(
            self.centering(L_X)
            * self.centering(L_Y)
        )

    def linear_CKA(self, X, Y):

        hsic = self.linear_HSIC(X, Y)

        var1 = torch.sqrt(
            self.linear_HSIC(X, X)
        )

        var2 = torch.sqrt(
            self.linear_HSIC(Y, Y)
        )

        return hsic / (var1 * var2)


# ============================================================
# Activation loading
# ============================================================

def find_layers(model_dir):
    """
    Find available layer numbers from files such as:
        layer_0_0.pt
        layer_1_0.pt
        ...
    """

    layers = set()

    for path in model_dir.glob("layer_*_*.pt"):

        parts = path.stem.split("_")

        if len(parts) >= 3:
            layers.add(int(parts[1]))

    return sorted(layers)


def load_layer_acts(
    model_dir,
    layer,
    device,
    center=True,
    scale=True,
):
    """
    Load all activation chunks belonging to one layer.
    """

    files = list(
        model_dir.glob(
            f"layer_{layer}_*.pt"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No activation files found for "
            f"layer {layer} in {model_dir}"
        )

    # Sort using the final batch index
    files = sorted(
        files,
        key=lambda p: int(
            p.stem.split("_")[-1]
        ),
    )

    activations = [
        torch.load(
            file,
            map_location="cpu",
        )
        for file in files
    ]

    acts = torch.cat(
        activations,
        dim=0,
    ).float()

    acts = acts.to(device)

    # Match original REEF preprocessing
    if center:
        acts = (
            acts
            - torch.mean(
                acts,
                dim=0,
            )
        )

    if scale:

        std = torch.std(
            acts,
            dim=0,
        )

        acts = (
            acts
            / std.clamp_min(1e-8)
        )

    return acts



def load_metadata(model_dir):

    metadata_path = (
        model_dir / "metadata.json"
    )

    if not metadata_path.exists():
        return None

    return json.loads(
        metadata_path.read_text()
    )


def validate_metadata(
    base_metadata,
    test_metadata,
):
    """
    Check that both fingerprints were generated using
    equivalent evaluation inputs.
    """

    if (
        base_metadata is None
        or test_metadata is None
    ):
        return

    checks = [
        "sample_count",
        "token_position",
        "dataset_sha256",
    ]

    for key in checks:

        if (
            base_metadata.get(key)
            != test_metadata.get(key)
        ):
            raise ValueError(
                f"Activation metadata mismatch "
                f"for '{key}': "
                f"{base_metadata.get(key)} != "
                f"{test_metadata.get(key)}"
            )


# ============================================================
# Evaluation
# ============================================================

def compute_cka_similarity(
    base_dir,
    test_dir,
    device,
):

    base_layers = find_layers(
        base_dir
    )

    test_layers = find_layers(
        test_dir
    )

    if not base_layers:
        raise ValueError(
            f"No activation layers found in "
            f"{base_dir}"
        )

    if not test_layers:
        raise ValueError(
            f"No activation layers found in "
            f"{test_dir}"
        )

    print(
        f"Base layers: {base_layers}"
    )

    print(
        f"Test layers: {test_layers}"
    )

    # Corresponding layer numbers available
    # in BOTH models
    common_layers = sorted(
        set(base_layers)
        & set(test_layers)
    )

    if not common_layers:
        raise ValueError(
            "The two models have no common "
            "layer indices."
        )

    print(
        f"Corresponding layers: "
        f"{common_layers}"
    )

    cka = CudaCKA(device)

    diagonal_scores = []

    for layer in common_layers:

        X = load_layer_acts(
            base_dir,
            layer,
            device,
            center=True,
            scale=True,
        )

        Y = load_layer_acts(
            test_dir,
            layer,
            device,
            center=True,
            scale=True,
        )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch at "
                f"layer {layer}: "
                f"{X.shape[0]} vs "
                f"{Y.shape[0]}"
            )

        score = cka.linear_CKA(
            X,
            Y,
        )

        score = float(
            score.detach().cpu()
        )

        diagonal_scores.append(
            score
        )

        print(
            f"Layer {layer}: "
            f"CKA = {score:.6f}"
        )

    cka_sim = float(
        np.mean(
            diagonal_scores
        )
    )

    return (
        cka_sim,
        common_layers,
        diagonal_scores,
    )


# ============================================================
# Main
# ============================================================

def main():

    workspace = Path(
        os.environ.get(
            "WORKSPACE",
            "/workspace",
        )
    )

    input_dir = (
        workspace / "input"
    )

    output_dir = (
        workspace / "output"
    )

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    config_path = (
        input_dir / "config.json"
    )

    config = {}

    if config_path.exists():

        config = json.loads(
            config_path.read_text()
        )

    # Model folder names produced during deployment
    base_model = config.get(
        "base_model",
        "gpt2",
    )

    test_model = config.get(
        "test_model",
        "gpt2-imdb",
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    base_dir = (
        input_dir / base_model
    )

    test_dir = (
        input_dir / test_model
    )

    print("=" * 60)
    print("REEF CKA EVALUATION")
    print("=" * 60)

    print(
        f"Input directory: {input_dir}"
    )

    print(
        f"Base model:      {base_model}"
    )

    print(
        f"Test model:      {test_model}"
    )

    print(
        f"Device:          {device}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Verify activation directories
    # --------------------------------------------------------

    if not base_dir.exists():

        results = {
            "evaluator": "reef_cka",
            "success": False,
            "error": (
                f"Base activation directory "
                f"not found: {base_dir}"
            ),
            "metrics": {},
        }

        (
            output_dir
            / "evaluation_results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
        )

        return

    if not test_dir.exists():

        results = {
            "evaluator": "reef_cka",
            "success": False,
            "error": (
                f"Test activation directory "
                f"not found: {test_dir}"
            ),
            "metrics": {},
        }

        (
            output_dir
            / "evaluation_results.json"
        ).write_text(
            json.dumps(
                results,
                indent=2,
            )
        )

        return

    try:

        # ----------------------------------------------------
        # Validate metadata
        # ----------------------------------------------------

        base_metadata = (
            load_metadata(base_dir)
        )

        test_metadata = (
            load_metadata(test_dir)
        )

        validate_metadata(
            base_metadata,
            test_metadata,
        )

        # ----------------------------------------------------
        # CKA
        # ----------------------------------------------------

        (
            cka_sim,
            layers,
            layer_scores,
        ) = compute_cka_similarity(
            base_dir,
            test_dir,
            device,
        )

        metrics = {
            "cka_sim": cka_sim,
        }

        print()
        print(
            f"Mean diagonal CKA: "
            f"{cka_sim:.6f}"
        )

        results = {
            "evaluator": "reef_cka",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "details": {
                "base_model": base_model,
                "test_model": test_model,
                "layers": layers,
                "layer_cka": layer_scores,
            },
            "timestamp": (
                datetime.utcnow()
                .isoformat()
                + "Z"
            ),
        }

    except Exception as e:

        results = {
            "evaluator": "reef_cka",
            "success": False,
            "error": str(e),
            "metrics": {},
        }

    # --------------------------------------------------------
    # Save Landseer result
    # --------------------------------------------------------

    result_path = (
        output_dir
        / "evaluation_results.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print(
        f"Results saved to "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()