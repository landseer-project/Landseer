#!/usr/bin/env python3
"""Explanation faithfulness evaluator: mean SHAP Drop@10 score."""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def evaluate_faithfulness_drop10(drop10: np.ndarray) -> float:
    """
    Evaluate explanation faithfulness using Drop@10%.

    Args:
        drop10:
            Per-sample Drop@10 scores produced by the SHAP explanation tool.
            Higher is better.

    Returns:
        Mean Drop@10 score across samples.
    """

    print(f"Evaluating faithfulness Drop@10 with shape: {drop10.shape}")

    if drop10.ndim != 1:
        raise ValueError("drop10 must be a 1D array")

    if len(drop10) == 0:
        raise ValueError("drop10 array is empty")

    mean_drop10 = float(drop10.mean())

    return mean_drop10


def prepare_drop10_scores(drop10_data: np.ndarray) -> np.ndarray:
    """
    Convert loaded faith_drop10.npy data into a 1D array.

    Handles scalar, 1D, 2D, or higher-dimensional input.
    """

    print(
        f"Loaded faith_drop10.npy "
        f"shape={drop10_data.shape}, ndim={drop10_data.ndim}"
    )

    if drop10_data.ndim == 0:
        # Scalar
        drop10_scores = np.array([drop10_data.item()])

    elif drop10_data.ndim == 1:
        # Expected format
        drop10_scores = drop10_data

    elif drop10_data.ndim == 2:
        # Preserve behavior from previous evaluator:
        # use first row
        drop10_scores = drop10_data[0]

        print(
            "2D Drop@10 array detected. "
            f"Using first row with shape {drop10_scores.shape}"
        )

    else:
        # Higher dimensional
        drop10_scores = drop10_data.flatten()

        print(
            "Multi-dimensional Drop@10 array detected. "
            f"Flattened to shape {drop10_scores.shape}"
        )

    return np.asarray(drop10_scores).reshape(-1)


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))

    input_dir = workspace / "input"
    output_dir = workspace / "output"

    output_dir.mkdir(exist_ok=True, parents=True)

    faith_drop10_path = input_dir / "faith_drop10.npy"

    # ------------------------------------------------------
    # Check required explanation output
    # ------------------------------------------------------
    if not faith_drop10_path.exists():
        payload = {
            "evaluator": "explanations",
            "success": True,
            "skipped": True,
            "skip_reason": "faith_drop10.npy missing",
            "metrics": {
                "drop10_score": None
            },
        }

        (output_dir / "evaluation_results.json").write_text(
            json.dumps(payload, indent=2)
        )

        print(
            f"Skipping explanation evaluation: "
            f"{faith_drop10_path} not found"
        )

        return

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------
    try:
        print(f"Loading Drop@10 scores from: {faith_drop10_path}")

        drop10_data = np.load(faith_drop10_path)

        drop10_scores = prepare_drop10_scores(drop10_data)

        score = evaluate_faithfulness_drop10(drop10_scores)

        payload = {
            "evaluator": "explanations",
            "success": True,
            "skipped": False,
            "metrics": {
                "drop10_score": score
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        print(f"drop10_score: {score:.6f}")

    except Exception as exc:
        payload = {
            "evaluator": "explanations",
            "success": False,
            "error": str(exc),
            "metrics": {},
        }

        print(f"Explanation evaluation failed: {exc}")

    # ------------------------------------------------------
    # Save standard pipeline result
    # ------------------------------------------------------
    results_path = output_dir / "evaluation_results.json"

    results_path.write_text(
        json.dumps(payload, indent=2)
    )

    print(f"Saved evaluation results to: {results_path}")


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# """Explanation fidelity: Drop10 (accuracy drop after zeroing top-10% gradient pixels)."""

# import json
# import os
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset

# from model_loader import load_torch_model_for_eval


# def _accuracy(model, loader, device):
#     model.eval()
#     correct = total = 0
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             correct += (model(x).argmax(1) == y).sum().item()
#             total += y.size(0)
#     return correct / total if total else 0.0


# def drop10_score(model, x, y, device, frac=0.1, max_samples=128):
#     """Mean accuracy drop after zeroing top-frac |grad| pixels per sample."""
#     n = min(len(x), max_samples)
#     x = torch.tensor(x[:n]).float().to(device)
#     y = torch.tensor(y[:n]).long().to(device)

#     model.eval()
#     x_req = x.clone().detach().requires_grad_(True)
#     loss = torch.nn.functional.cross_entropy(model(x_req), y)
#     loss.backward()
#     grads = x_req.grad.detach().abs()

#     flat = grads.view(n, -1)
#     k = max(1, int(flat.shape[1] * frac))
#     topk = torch.topk(flat, k, dim=1).indices
#     mask = torch.ones_like(flat)
#     mask.scatter_(1, topk, 0.0)
#     x_drop = (x.view(n, -1) * mask).view_as(x)

#     loader_clean = DataLoader(TensorDataset(x.detach().cpu(), y.cpu()), batch_size=32)
#     loader_drop = DataLoader(TensorDataset(x_drop.detach().cpu(), y.cpu()), batch_size=32)
#     acc_clean = _accuracy(model, loader_clean, device)
#     acc_drop = _accuracy(model, loader_drop, device)
#     return float(acc_clean - acc_drop)


# def main():
#     workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
#     input_dir = workspace / "input"
#     output_dir = workspace / "output"
#     output_dir.mkdir(exist_ok=True, parents=True)

#     model_path = input_dir / "model.pt"
#     test_data = input_dir / "test_data.npy"
#     test_labels = input_dir / "test_labels.npy"
#     if not model_path.exists() or not test_data.exists() or not test_labels.exists():
#         payload = {
#             "evaluator": "explanations",
#             "success": True,
#             "skipped": True,
#             "skip_reason": "model.pt or test data missing",
#             "metrics": {"drop10_score": None},
#         }
#         (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))
#         return

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     try:
#         model = load_torch_model_for_eval(model_path, input_dir, device)
#         score = drop10_score(model, np.load(test_data), np.load(test_labels), device)
#         payload = {
#             "evaluator": "explanations",
#             "success": True,
#             "skipped": False,
#             "metrics": {"drop10_score": score},
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#         }
#         print(f"drop10_score: {score:.4f}")
#     except Exception as exc:
#         payload = {
#             "evaluator": "explanations",
#             "success": False,
#             "error": str(exc),
#             "metrics": {},
#         }

#     (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))


# if __name__ == "__main__":
#     main()
