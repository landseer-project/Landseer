#!/usr/bin/env python3
"""
LiRA Membership Inference Evaluator for Landseer

This evaluator:
1. Loads target model from /workspace/input/model.pt
2. Loads member data from /workspace/input/data.npy and labels.npy
3. Loads non-member data from /workspace/input/test_data.npy and test_labels.npy
4. Computes target scores on data.npy + test_data.npy
5. Loads saved non-DP shadow keep/scores
6. Computes Online LiRA AUC
7. Writes /workspace/output/evaluation_results.json

Expected input files:
    input/model.pt
    input/data.npy
    input/labels.npy
    input/test_data.npy
    input/test_labels.npy
    input/celeba_shadows/
        model_000/
            keep.npy
            scores.npy
        model_001/
            keep.npy
            scores.npy
        ...
        model_127/
            keep.npy
            scores.npy
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

from model_loader import load_torch_model_for_eval


# ============================================================
# EDIT THESE SETTINGS
# ============================================================

# Folder containing ONLY shadow models model_000 to model_127.
SHADOW_SCORES_SUBDIR = "celeba_shadows"


# Must match how the target model and shadow models were trained.
# For our CelebA ResNet20 code, we used "half" by default:
#     x -> (x - 0.5) / 0.5
NORMALIZE_MODE = "none"   # choices: "none", "half", "imagenet", "cifar10"

# ResNet20 CelebA setup used 32x32 images.
IMAGE_SIZE = 32

BATCH_SIZE = 256
NUM_WORKERS = 4

# Online LiRA setting.
FIX_VARIANCE = False

# Save computed target files in output/
SAVE_TARGET_FILES = True


# ============================================================
# Dataset
# ============================================================

class NPYImageDataset(Dataset):
    """
    Dataset for .npy image arrays and labels.

    Supports:
        NCHW: (N, C, H, W)
        NHWC: (N, H, W, C)

    Labels:
        0/1 for CelebA binary attribute
        {-1, 1} will be converted to {0, 1}
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, normalize_mode: str = "half"):
        if X.ndim != 4:
            raise ValueError(f"Expected X.ndim == 4, got shape {X.shape}")

        y = np.asarray(y).reshape(-1)

        if len(X) != len(y):
            raise ValueError(f"N mismatch: X has {len(X)}, y has {len(y)}")

        unique_y = set(np.unique(y).tolist())
        if unique_y.issubset({-1, 1}):
            y = ((y + 1) // 2).astype(np.int64)

        self.X = X
        self.y = y.astype(np.int64)
        self.normalize_mode = normalize_mode

        if X.shape[1] in [1, 3]:
            self.layout = "NCHW"
        elif X.shape[-1] in [1, 3]:
            self.layout = "NHWC"
        else:
            raise ValueError(
                f"Could not infer image layout from shape {X.shape}. "
                "Expected NCHW or NHWC with 1 or 3 channels."
            )

    def __len__(self):
        return int(len(self.y))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "none":
            return x

        if self.normalize_mode == "half":
            mean = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], dtype=x.dtype).view(3, 1, 1)
            return (x - mean) / std

        if self.normalize_mode == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
            return (x - mean) / std

        if self.normalize_mode == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=x.dtype).view(3, 1, 1)
            return (x - mean) / std

        raise ValueError(f"Unknown NORMALIZE_MODE: {self.normalize_mode}")

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()

        if self.layout == "NHWC":
            x = x.permute(2, 0, 1).contiguous()

        # Convert uint8 / 0-255 floats to [0, 1]
        if x.max() > 2.0:
            x = x / 255.0

        # Convert grayscale to RGB if needed.
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)

        x = self._normalize(x)
        y = int(self.y[idx])

        return x, y


# ============================================================
# Utility
# ============================================================

def write_results(output_dir: Path, results: dict):
    output_dir.mkdir(exist_ok=True, parents=True)
    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {results_path}")


def load_train_test_arrays(input_dir: Path):
    required = [
        "data.npy",
        "labels.npy",
        "test_data.npy",
        "test_labels.npy",
    ]

    for name in required:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    X_train = np.load(input_dir / "data.npy")
    y_train = np.load(input_dir / "labels.npy")
    X_test = np.load(input_dir / "test_data.npy")
    y_test = np.load(input_dir / "test_labels.npy")

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    return X_train, y_train, X_test, y_test


@torch.no_grad()
def compute_true_class_logprob_scores(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> np.ndarray:
    """
    Compute true-class log probability for each audit example.

    Output:
        scores shape = (N, 1)
    """

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=("cuda" in str(device)),
    )

    all_scores = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if X_batch.shape[-2:] != (image_size, image_size):
            X_batch = F.interpolate(
                X_batch,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )

        logits = model(X_batch)

        if int(y_batch.max()) >= logits.shape[1]:
            raise ValueError(
                f"Label {int(y_batch.max())} is out of range for model output "
                f"with {logits.shape[1]} classes."
            )

        log_probs = F.log_softmax(logits, dim=1)

        true_class_logprob = log_probs[
            torch.arange(y_batch.size(0), device=device),
            y_batch,
        ]

        all_scores.append(true_class_logprob.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0).reshape(-1, 1)
    return scores


def load_shadow_keep_scores(shadow_scores_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all shadow folders inside shadow_scores_dir.

    This expects shadow_scores_dir to contain ONLY shadow models:
        model_000 ... model_127

    Each folder must contain:
        keep.npy
        scores.npy
    """

    if not shadow_scores_dir.exists():
        raise FileNotFoundError(f"SHADOW_SCORES_DIR does not exist: {shadow_scores_dir}")

    model_dirs = []

    for name in sorted(os.listdir(shadow_scores_dir)):
        path = shadow_scores_dir / name
        if not path.is_dir():
            continue

        keep_path = path / "keep.npy"
        scores_path = path / "scores.npy"

        if keep_path.exists() and scores_path.exists():
            model_dirs.append(path)

    if len(model_dirs) == 0:
        raise ValueError(f"No valid shadow folders found in {shadow_scores_dir}")

    keep_list = []
    scores_list = []

    for path in model_dirs:
        keep_list.append(np.load(path / "keep.npy"))
        scores_list.append(np.load(path / "scores.npy"))

    shadow_keep = np.asarray(keep_list).astype(bool)
    shadow_scores = np.asarray(scores_list).astype(np.float64)

    if shadow_keep.ndim != 2:
        raise ValueError(f"Expected shadow_keep ndim 2, got {shadow_keep.shape}")

    if shadow_scores.ndim != 3:
        raise ValueError(f"Expected shadow_scores ndim 3, got {shadow_scores.shape}")

    if shadow_keep.shape[0] != shadow_scores.shape[0]:
        raise ValueError(
            f"Shadow model count mismatch: keep {shadow_keep.shape}, "
            f"scores {shadow_scores.shape}"
        )

    if shadow_keep.shape[1] != shadow_scores.shape[1]:
        raise ValueError(
            f"Shadow audit-size mismatch: keep {shadow_keep.shape}, "
            f"scores {shadow_scores.shape}"
        )

    return shadow_keep, shadow_scores, [str(p) for p in model_dirs]


# ============================================================
# LiRA
# ============================================================

def online_lira_predictions(
    shadow_keep: np.ndarray,
    shadow_scores: np.ndarray,
    target_keep: np.ndarray,
    target_scores: np.ndarray,
    fix_variance: bool = False,
):
    """
    Online LiRA attack.

    shadow_keep:
        shape = (num_shadow_models, audit_size)

    shadow_scores:
        shape = (num_shadow_models, audit_size, 1)

    target_keep:
        shape = (audit_size,)

    target_scores:
        shape = (audit_size, 1)
    """

    if target_keep.ndim == 1:
        target_keep = target_keep[None, :]

    if target_scores.ndim == 2:
        target_scores = target_scores[None, :, :]

    audit_size = shadow_scores.shape[1]

    if shadow_keep.shape[1] != audit_size:
        raise ValueError("shadow_keep and shadow_scores audit size mismatch")

    if target_keep.shape[1] != audit_size:
        raise ValueError(
            f"target_keep audit size {target_keep.shape[1]} != shadow audit size {audit_size}"
        )

    if target_scores.shape[1] != audit_size:
        raise ValueError(
            f"target_scores audit size {target_scores.shape[1]} != shadow audit size {audit_size}"
        )

    dat_in = []
    dat_out = []
    valid_indices = []

    for j in range(audit_size):
        in_j = shadow_scores[shadow_keep[:, j], j, :]
        out_j = shadow_scores[~shadow_keep[:, j], j, :]

        if len(in_j) == 0 or len(out_j) == 0:
            continue

        dat_in.append(in_j)
        dat_out.append(out_j)
        valid_indices.append(j)

    if len(valid_indices) == 0:
        raise ValueError("No valid audit examples with both IN and OUT shadow scores.")

    valid_indices = np.asarray(valid_indices)

    target_keep = target_keep[:, valid_indices]
    target_scores = target_scores[:, valid_indices, :]

    in_size = min(map(len, dat_in))
    out_size = min(map(len, dat_out))

    dat_in = np.asarray([x[:in_size] for x in dat_in])
    dat_out = np.asarray([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, axis=1)
    mean_out = np.median(dat_out, axis=1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_out)
    else:
        std_in = np.std(dat_in, axis=1)
        std_out = np.std(dat_out, axis=1)

    prediction_scores = []
    membership_answers = []

    for answer, target_score in zip(target_keep, target_scores):
        pr_in = -scipy.stats.norm.logpdf(
            target_score,
            mean_in,
            std_in + 1e-30,
        )

        pr_out = -scipy.stats.norm.logpdf(
            target_score,
            mean_out,
            std_out + 1e-30,
        )

        lira_score = pr_in - pr_out

        prediction_scores.extend(lira_score.mean(axis=1))
        membership_answers.extend(answer)

    return (
        np.asarray(prediction_scores),
        np.asarray(membership_answers, dtype=bool),
        int(len(valid_indices)),
        int(audit_size),
    )


def compute_auc_metrics(prediction_scores: np.ndarray, membership_answers: np.ndarray):
    """
    Match our previous evaluator convention:

        roc_curve(membership_answers, -prediction_scores)
    """

    if len(np.unique(membership_answers)) < 2:
        raise ValueError(
            "Cannot compute MIA AUC because target_keep has only one class. "
            "The target must have both members and non-members."
        )

    fpr, tpr, _ = roc_curve(membership_answers, -prediction_scores)

    mia_auc = float(auc(fpr, tpr))
    mia_acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))

    low_fpr_idx = np.where(fpr < 0.0001)[0]
    if len(low_fpr_idx) > 0:
        tpr_at_001_percent_fpr = float(tpr[low_fpr_idx[-1]])
    else:
        tpr_at_001_percent_fpr = 0.0

    return mia_auc, mia_acc, tpr_at_001_percent_fpr


# ============================================================
# Main
# ============================================================

def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir = workspace / "input"
    output_dir = workspace / "output"
    shadow_scores_dir = input_dir / SHADOW_SCORES_SUBDIR
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Workspace:", workspace)
    print("Input dir:", input_dir)
    print("Output dir:", output_dir)
    print("Device:", device)
    print("Shadow scores dir:", shadow_scores_dir)
    print("Normalize mode:", NORMALIZE_MODE)
    print("Image size:", IMAGE_SIZE)

    model_path = input_dir / "model.pt"

    if not model_path.exists():
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": "model.pt not found in input directory",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    # ------------------------------------------------------------
    # Load target model
    # ------------------------------------------------------------
    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
        model.eval()
    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": f"Failed to load model: {e}",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    # ------------------------------------------------------------
    # Load target audit data
    # ------------------------------------------------------------
    try:
        X_train, y_train, X_test, y_test = load_train_test_arrays(input_dir)
    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": f"Failed to load input npy files: {e}",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    n_train = int(len(X_train))
    n_test = int(len(X_test))
    audit_size = n_train + n_test

    print("Loaded member train examples:", n_train)
    print("Loaded non-member test examples:", n_test)
    print("Target audit size:", audit_size)

    X_audit = np.concatenate([X_train, X_test], axis=0)
    y_audit = np.concatenate(
        [
            np.asarray(y_train).reshape(-1),
            np.asarray(y_test).reshape(-1),
        ],
        axis=0,
    )

    target_keep = np.concatenate(
        [
            np.ones(n_train, dtype=bool),
            np.zeros(n_test, dtype=bool),
        ],
        axis=0,
    )

    # ------------------------------------------------------------
    # Compute target scores
    # ------------------------------------------------------------
    try:
        audit_dataset = NPYImageDataset(
            X_audit,
            y_audit,
            normalize_mode=NORMALIZE_MODE,
        )

        target_scores = compute_true_class_logprob_scores(
            model=model,
            dataset=audit_dataset,
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            image_size=IMAGE_SIZE,
        )

    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": f"Failed to compute target scores: {e}",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    if target_scores.shape != (audit_size, 1):
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": (
                f"Bad target_scores shape: {target_scores.shape}, "
                f"expected {(audit_size, 1)}"
            ),
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    if SAVE_TARGET_FILES:
        np.save(output_dir / "target_scores.npy", target_scores)
        np.save(output_dir / "target_keep.npy", target_keep)

    # ------------------------------------------------------------
    # Load shadow scores
    # ------------------------------------------------------------
    try:
        shadow_keep, shadow_scores, loaded_shadow_dirs = load_shadow_keep_scores(
            shadow_scores_dir
        )
    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": f"Failed to load shadow scores: {e}",
            "metrics": {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    print("Loaded shadow models:", shadow_keep.shape[0])
    print("Shadow keep shape:", shadow_keep.shape)
    print("Shadow scores shape:", shadow_scores.shape)
    print("Target keep shape:", target_keep.shape)
    print("Target scores shape:", target_scores.shape)

    if shadow_keep.shape[1] != audit_size:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": (
                f"Shadow audit size {shadow_keep.shape[1]} does not match "
                f"target audit size {audit_size}. "
                "Shadow and target must use the same audit order: "
                "data.npy followed by test_data.npy."
            ),
            "metrics": {},
            "parameters": {
                "shadow_scores_dir": str(shadow_scores_dir),
                "shadow_keep_shape": str(shadow_keep.shape),
                "shadow_scores_shape": str(shadow_scores.shape),
                "target_scores_shape": str(target_scores.shape),
                "target_keep_shape": str(target_keep.shape),
                "target_audit_size": audit_size,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        write_results(output_dir, results)
        return

    # ------------------------------------------------------------
    # Run Online LiRA
    # ------------------------------------------------------------
    try:
        prediction_scores, membership_answers, valid_n, total_n = online_lira_predictions(
            shadow_keep=shadow_keep,
            shadow_scores=shadow_scores,
            target_keep=target_keep,
            target_scores=target_scores,
            fix_variance=FIX_VARIANCE,
        )

        mia_auc, mia_acc, tpr_low = compute_auc_metrics(
            prediction_scores,
            membership_answers,
        )

        metrics = {
            "mia_auc": float(mia_auc),
            "mia_online_auc": float(mia_auc),
            "mia_online_accuracy": float(mia_acc),
            "mia_tpr_at_0.01pct_fpr": float(tpr_low),
        }

        print(f"MIA Online LiRA AUC: {mia_auc:.6f}")
        print(f"MIA Online LiRA Accuracy: {mia_acc:.6f}")
        print(f"TPR@0.01%FPR: {tpr_low:.6f}")

        results = {
            "evaluator": "mia_lira",
            "success": True,
            "skipped": False,
            "metrics": metrics,
            "parameters": {
                "shadow_scores_dir": str(shadow_scores_dir),
                "num_shadow_models": int(shadow_keep.shape[0]),
                "normalize_mode": NORMALIZE_MODE,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "fix_variance": FIX_VARIANCE,
                "n_members_from_data_npy": n_train,
                "n_nonmembers_from_test_data_npy": n_test,
                "audit_size": audit_size,
                "valid_audit_examples": valid_n,
                "total_audit_examples": total_n,
                "target_keep_definition": "data.npy=members, test_data.npy=nonmembers",
                "shadow_audit_order_required": "same as target: data.npy followed by test_data.npy",
                "loaded_shadow_dirs_count": len(loaded_shadow_dirs),
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": str(e),
            "metrics": {},
            "parameters": {
                "shadow_scores_dir": str(shadow_scores_dir),
                "target_audit_size": audit_size,
                "target_keep_definition": "data.npy=members, test_data.npy=nonmembers",
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    write_results(output_dir, results)


if __name__ == "__main__":
    main()