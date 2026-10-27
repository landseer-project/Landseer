#!/usr/bin/env python3
"""Dataset-generic LiRA membership inference evaluator for Landseer."""

import json, os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

from model_loader import load_torch_model_for_eval


BATCH_SIZE = 256
NUM_WORKERS = 4
FIX_VARIANCE = False
SAVE_TARGET_FILES = True

DATASET_CONFIGS = {
    "celeba": {
        "normalize_mode": "half",
        "image_size": 32,
        "shadow_dir": "celeba_shadows",
        "expected_audit_size": 202599,
    },
    "cifar10": {
        "normalize_mode": "cifar10",
        "image_size": 32,
        "shadow_dir": "cifar10_shadows",
        "expected_audit_size": 60000,
    },
}


def write_results(output_dir: Path, results: dict):
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / "evaluation_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {path}")


def get_dataset_config():
    dataset = os.environ.get("LANDSEER_DATASET", "").strip().lower()
    dataset = {"cifar-10": "cifar10", "cifar_10": "cifar10"}.get(dataset, dataset)
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported LANDSEER_DATASET={dataset!r}; supported={list(DATASET_CONFIGS)}")
    return dataset, DATASET_CONFIGS[dataset]


def resolve_shadow_dir(input_dir: Path, cfg: dict) -> Path:
    artifact_dir = input_dir / "privacy"
    return artifact_dir / cfg["shadow_dir"]


class NPYImageDataset(Dataset):
    def __init__(self, X, y, normalize_mode="none"):
        if X.ndim != 4:
            raise ValueError(f"Expected 4D image array, got {X.shape}")

        y = np.asarray(y).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"N mismatch: X={len(X)}, y={len(y)}")

        if set(np.unique(y).tolist()).issubset({-1, 1}):
            y = ((y + 1) // 2).astype(np.int64)

        self.X = X
        self.y = y.astype(np.int64)
        self.normalize_mode = normalize_mode

        if X.shape[1] in [1, 3]:
            self.layout = "NCHW"
        elif X.shape[-1] in [1, 3]:
            self.layout = "NHWC"
        else:
            raise ValueError(f"Could not infer NCHW/NHWC layout from {X.shape}")

    def __len__(self):
        return len(self.y)

    def _normalize(self, x):
        if self.normalize_mode == "none":
            return x
        if self.normalize_mode == "half":
            mean, std = [0.5] * 3, [0.5] * 3
        elif self.normalize_mode == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif self.normalize_mode == "cifar10":
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        else:
            raise ValueError(f"Unknown normalize_mode={self.normalize_mode!r}")

        mean = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
        return (x - mean) / std

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        if self.layout == "NHWC":
            x = x.permute(2, 0, 1).contiguous()
        if x.max() > 2:
            x = x / 255.0
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return self._normalize(x), int(self.y[idx])


def load_train_test_arrays(input_dir: Path):
    required = ["data.npy", "labels.npy", "test_data.npy", "test_labels.npy"]
    for name in required:
        if not (input_dir / name).exists():
            raise FileNotFoundError(f"Required file not found: {input_dir / name}")

    return (
        np.load(input_dir / "data.npy"),
        np.load(input_dir / "labels.npy").reshape(-1),
        np.load(input_dir / "test_data.npy"),
        np.load(input_dir / "test_labels.npy").reshape(-1),
    )


@torch.no_grad()
def compute_true_class_logprob_scores(model, dataset, device, batch_size, num_workers, image_size):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=("cuda" in str(device))
    )

    scores = []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if X.shape[-2:] != (image_size, image_size):
            X = F.interpolate(X, size=(image_size, image_size), mode="bilinear", align_corners=False)

        logits = model(X)
        if int(y.max()) >= logits.shape[1]:
            raise ValueError(f"Label {int(y.max())} out of range for {logits.shape[1]} model classes")

        log_probs = F.log_softmax(logits, dim=1)
        scores.append(log_probs[torch.arange(y.size(0), device=device), y].cpu().numpy())

    return np.concatenate(scores).reshape(-1, 1)


def load_shadow_keep_scores(shadow_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not shadow_dir.exists():
        raise FileNotFoundError(f"Shadow directory not found: {shadow_dir}")

    model_dirs = []
    for name in sorted(os.listdir(shadow_dir)):
        path = shadow_dir / name
        if path.is_dir() and (path / "keep.npy").exists() and (path / "scores.npy").exists():
            model_dirs.append(path)

    if not model_dirs:
        raise ValueError(f"No valid shadow models found in {shadow_dir}")

    keep = np.asarray([np.load(p / "keep.npy") for p in model_dirs]).astype(bool)
    scores = np.asarray([np.load(p / "scores.npy") for p in model_dirs]).astype(np.float64)

    if keep.ndim != 2 or scores.ndim != 3:
        raise ValueError(f"Bad shadow shapes: keep={keep.shape}, scores={scores.shape}")
    if keep.shape[:2] != scores.shape[:2]:
        raise ValueError(f"Shadow keep/scores mismatch: {keep.shape} vs {scores.shape}")

    return keep, scores, [str(p) for p in model_dirs]


def online_lira_predictions(shadow_keep, shadow_scores, target_keep, target_scores, fix_variance=False):
    if target_keep.ndim == 1:
        target_keep = target_keep[None, :]
    if target_scores.ndim == 2:
        target_scores = target_scores[None, :, :]

    audit_size = shadow_scores.shape[1]
    if shadow_keep.shape[1] != audit_size or target_keep.shape[1] != audit_size or target_scores.shape[1] != audit_size:
        raise ValueError("Target/shadow audit-size mismatch")

    dat_in, dat_out, valid = [], [], []
    for j in range(audit_size):
        in_j = shadow_scores[shadow_keep[:, j], j, :]
        out_j = shadow_scores[~shadow_keep[:, j], j, :]
        if len(in_j) and len(out_j):
            dat_in.append(in_j)
            dat_out.append(out_j)
            valid.append(j)

    if not valid:
        raise ValueError("No audit examples have both IN and OUT shadow scores")

    valid = np.asarray(valid)
    target_keep = target_keep[:, valid]
    target_scores = target_scores[:, valid, :]

    in_size, out_size = min(map(len, dat_in)), min(map(len, dat_out))
    dat_in = np.asarray([x[:in_size] for x in dat_in])
    dat_out = np.asarray([x[:out_size] for x in dat_out])

    mean_in, mean_out = np.median(dat_in, axis=1), np.median(dat_out, axis=1)
    if fix_variance:
        std_in, std_out = np.std(dat_in), np.std(dat_out)
    else:
        std_in, std_out = np.std(dat_in, axis=1), np.std(dat_out, axis=1)

    prediction_scores, answers = [], []
    for answer, target_score in zip(target_keep, target_scores):
        pr_in = -scipy.stats.norm.logpdf(target_score, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(target_score, mean_out, std_out + 1e-30)
        prediction_scores.extend((pr_in - pr_out).mean(axis=1))
        answers.extend(answer)

    return np.asarray(prediction_scores), np.asarray(answers, dtype=bool), len(valid), audit_size


def compute_auc_metrics(prediction_scores, membership_answers):
    if len(np.unique(membership_answers)) < 2:
        raise ValueError("Target membership contains only one class")

    fpr, tpr, _ = roc_curve(membership_answers, -prediction_scores)
    mia_auc = float(auc(fpr, tpr))
    mia_acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    idx = np.where(fpr < 0.0001)[0]
    tpr_low = float(tpr[idx[-1]]) if len(idx) else 0.0
    return mia_auc, mia_acc, tpr_low


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    input_dir, output_dir = workspace / "input", workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dataset, cfg = get_dataset_config()
        normalize_mode = cfg["normalize_mode"]
        image_size = cfg["image_size"]
        expected_audit_size = cfg["expected_audit_size"]
        shadow_dir = resolve_shadow_dir(input_dir, cfg)
    except Exception as e:
        write_results(output_dir, {
            "evaluator": "mia_lira", "success": False,
            "error": f"Dataset configuration error: {e}", "metrics": {}
        })
        return

    print(f"Device: {device}")
    print(f"LANDSEER_DATASET: {dataset}")
    print(f"Shadow scores dir: {shadow_dir}")
    print(f"Normalize mode: {normalize_mode}")
    print(f"Image size: {image_size}")
    print(f"Expected audit size: {expected_audit_size}")

    model_path = input_dir / "model.pt"
    if not model_path.exists():
        write_results(output_dir, {
            "evaluator": "mia_lira", "success": False,
            "error": "model.pt not found", "metrics": {}
        })
        return

    try:
        model = load_torch_model_for_eval(model_path, input_dir, device)
        model.eval()

        X_train, y_train, X_test, y_test = load_train_test_arrays(input_dir)
        n_train, n_test = len(X_train), len(X_test)
        audit_size = n_train + n_test

        print(f"Members: {n_train}")
        print(f"Non-members: {n_test}")
        print(f"Target audit size: {audit_size}")

        if audit_size != expected_audit_size:
            raise ValueError(
                f"{dataset} audit size is {audit_size}, expected {expected_audit_size}. "
                "Target data must match the shadow-model audit universe."
            )

        X_audit = np.concatenate([X_train, X_test])
        y_audit = np.concatenate([y_train, y_test])
        target_keep = np.concatenate([
            np.ones(n_train, dtype=bool),
            np.zeros(n_test, dtype=bool)
        ])

        audit_dataset = NPYImageDataset(X_audit, y_audit, normalize_mode)
        target_scores = compute_true_class_logprob_scores(
            model, audit_dataset, device, BATCH_SIZE, NUM_WORKERS, image_size
        )

        shadow_keep, shadow_scores, loaded_dirs = load_shadow_keep_scores(shadow_dir)

        print(f"Loaded shadow models: {shadow_keep.shape[0]}")
        print(f"Shadow keep shape: {shadow_keep.shape}")
        print(f"Shadow scores shape: {shadow_scores.shape}")
        print(f"Target scores shape: {target_scores.shape}")

        if shadow_keep.shape[1] != audit_size:
            raise ValueError(
                f"Shadow audit size {shadow_keep.shape[1]} != target audit size {audit_size}. "
                "Audit order must be data.npy followed by test_data.npy."
            )

        if SAVE_TARGET_FILES:
            np.save(output_dir / "target_scores.npy", target_scores)
            np.save(output_dir / "target_keep.npy", target_keep)

        prediction_scores, answers, valid_n, total_n = online_lira_predictions(
            shadow_keep, shadow_scores, target_keep, target_scores, FIX_VARIANCE
        )

        mia_auc, mia_acc, tpr_low = compute_auc_metrics(prediction_scores, answers)

        metrics = {
            "mia_auc": mia_auc,
            "mia_online_auc": mia_auc,
            "mia_online_accuracy": mia_acc,
            "mia_tpr_at_0.01pct_fpr": tpr_low,
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
                "dataset": dataset,
                "shadow_scores_dir": str(shadow_dir),
                "shadow_model_set": cfg["shadow_dir"],
                "num_shadow_models": int(shadow_keep.shape[0]),
                "normalize_mode": normalize_mode,
                "image_size": image_size,
                "expected_audit_size": expected_audit_size,
                "n_members_from_data_npy": n_train,
                "n_nonmembers_from_test_data_npy": n_test,
                "audit_size": audit_size,
                "valid_audit_examples": valid_n,
                "total_audit_examples": total_n,
                "loaded_shadow_dirs_count": len(loaded_dirs),
                "target_keep_definition": "data.npy=members, test_data.npy=nonmembers",
                "shadow_audit_order_required": "data.npy followed by test_data.npy",
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        results = {
            "evaluator": "mia_lira",
            "success": False,
            "error": str(e),
            "metrics": {},
            "parameters": {"dataset": dataset, "shadow_scores_dir": str(shadow_dir)},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    write_results(output_dir, results)


if __name__ == "__main__":
    main()