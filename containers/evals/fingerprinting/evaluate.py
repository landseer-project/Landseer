#!/usr/bin/env python3
"""Dataset Inference Fingerprinting Evaluator.

Primary metric:
    fingerprinting = one-sided Welch p-value, lower is better.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model_loader import load_torch_model_for_eval
from fingerprinting import evaluate_fingerprinting


def write_result(output_dir, result):
    path = output_dir / "evaluation_results.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {path}")


# def resolve_calibrator(input_dir):
#     """Select the fixed Dataset Inference calibrator."""
#     explicit = os.environ.get("FINGERPRINTING_CALIBRATOR", "").strip()
#     if explicit:
#         path = Path(explicit)
#         return path if path.is_absolute() else input_dir / path

#     dataset = os.environ.get("LANDSEER_DATASET", "").strip().lower()
#     variant = os.environ.get("LANDSEER_DATASET_VARIANT", "").strip().lower()

#     if dataset == "celeba":
#         if variant == "clean":
#             return input_dir / "celeba_clean_calibrator_final.pt"
#         if variant in {"poisoned", "backdoor", "poison"}:
#             return input_dir / "celeba_backdoor_calibrator_final.pt"

#     raise ValueError(
#         f"Could not determine fingerprinting calibrator. "
#         f"LANDSEER_DATASET={dataset!r}, LANDSEER_DATASET_VARIANT={variant!r}. "
#         "Set those variables or FINGERPRINTING_CALIBRATOR."
#     )

def resolve_calibrator(input_dir):
    artifact_dir = input_dir / "fingerprinting"

    explicit = os.environ.get("FINGERPRINTING_CALIBRATOR", "").strip()
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else artifact_dir / path

    dataset = os.environ.get("LANDSEER_DATASET", "").strip().lower()
    variant = os.environ.get("LANDSEER_DATASET_VARIANT", "").strip().lower()

    if dataset == "celeba":
        if variant == "clean":
            return artifact_dir / "celeba_clean_calibrator_final.pt"
        if variant in {"poisoned", "backdoor", "poison"}:
            return artifact_dir / "celeba_backdoor_calibrator_final.pt"

    if dataset == "cifar10":
        return artifact_dir / "cifar10_backdoor_calibrator_final.pt"

    raise ValueError(
        f"No fingerprinting calibrator for dataset={dataset!r}, variant={variant!r}"
    )


def build_reference_loaders(calibrator, batch_size):
    required = ["reference_member_x", "reference_member_y", "reference_public_x", "reference_public_y"]
    missing = [k for k in required if k not in calibrator]
    if missing:
        raise ValueError("Calibrator missing reference data: " + ", ".join(missing))

    member = TensorDataset(calibrator["reference_member_x"].float(), calibrator["reference_member_y"].long())
    public = TensorDataset(calibrator["reference_public_x"].float(), calibrator["reference_public_y"].long())

    member_loader = DataLoader(member, batch_size=batch_size, shuffle=False)
    public_loader = DataLoader(public, batch_size=batch_size, shuffle=False)
    return member_loader, public_loader


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = input_dir / "model.pt"

    if not model_path.exists():
        write_result(output_dir, {"evaluator": "fingerprinting", "success": False, "error": "model.pt not found", "metrics": {}})
        return

    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
        calibrator_path = resolve_calibrator(input_dir)

        if not calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")

        calibrator = torch.load(calibrator_path, map_location="cpu", weights_only=False)
        batch_size = int(os.environ.get("FINGERPRINTING_BATCH_SIZE", calibrator.get("attack_batch_size", 100)))
        member_loader, public_loader = build_reference_loaders(calibrator, batch_size)

        print(f"Device: {device}")
        print(f"LANDSEER_DATASET: {os.environ.get('LANDSEER_DATASET')}")
        print(f"LANDSEER_DATASET_VARIANT: {os.environ.get('LANDSEER_DATASET_VARIANT')}")
        print(f"Calibrator: {calibrator_path}")
        print(f"Baseline: {calibrator.get('baseline_result')}")
        print(f"Reference member samples: {len(member_loader.dataset)}")
        print(f"Reference public samples: {len(public_loader.dataset)}")

        result = evaluate_fingerprinting(
            model=model,
            train_loader=member_loader,
            public_loader=public_loader,
            calibrator_path=calibrator_path,
            device=device,
        )

        metrics = {
            "fingerprinting": float(result["p_value"]),
            "fingerprinting_effect": float(result["mean_difference"]),
            "fingerprinting_evidence": float(result["evidence_score"]),
        }

        print(f"Fingerprint p-value: {metrics['fingerprinting']:.6e}")
        print(f"Fingerprint effect: {metrics['fingerprinting_effect']:.6f}")
        print(f"Fingerprint evidence: {metrics['fingerprinting_evidence']:.6f}")

        results = {
            "evaluator": "fingerprinting",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {
                "method": "dataset_inference_mingd",
                "calibrator": calibrator_path.name,
                "baseline_p_value": calibrator.get("baseline_result", {}).get("p_value"),
                "split_index": calibrator.get("split_index"),
                "pval_samples": calibrator.get("pval_samples"),
                "num_iter": calibrator.get("num_iter"),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        results = {"evaluator": "fingerprinting", "success": False, "error": str(e), "metrics": {}}

    write_result(output_dir, results)


if __name__ == "__main__":
    main()