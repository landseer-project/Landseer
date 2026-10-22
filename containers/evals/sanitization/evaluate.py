#!/usr/bin/env python3
"""
Sanitization Evaluator

Measures how well pre-training detectors separate poisoned from clean samples
on the training set the model was actually trained on.

Configured with ``input: training_input`` in evaluators.yaml, so /input is exactly
the input of the workflow's in-training task (dataset plus pre-training outputs).

Inputs
  data.npy                         training images the model was trained on
  detection_scores.<detector>.npz  written by a detector tool, one entry per row
                                   it scored: row_sha1 (SHA-1 of the row's bytes
                                   as stored in data.npy), score (higher = more
                                   suspicious), is_poisoned (ground truth)

Each training row is matched to the detector table by hash, so removals and
reordering by pre-training tools do not matter. Metrics per detector:
  <detector>_train_roc_auc, <detector>_train_precision_at_n
Precision@n follows get_precn in the XGBOD authors' repo (yzhao062/XGBOD).

A detector reports no metrics when a training row is missing from its table or
matches conflicting entries (data changed after scoring), or when the training
set has a single class; the evaluation is then reported as skipped with the
reason. A missing data.npy or detector table means a lost upstream output and
fails the evaluation.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

SCORES_GLOB = "detection_scores.*.npz"
_CHUNK = 8192


def roc_auc(y_true, score):
    """ROC-AUC via the rank-sum statistic with average ranks for ties (= sklearn roc_auc_score)."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(sorted_score)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(score)]))
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + 1 + e) / 2.0
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_at_n(y_true, score):
    """get_precn: n = number of true outliers; flag scores strictly above the matching
    percentile (scipy scoreatpercentile == numpy linear percentile)."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    threshold = np.percentile(score, 100 * (1 - y_true.sum() / len(y_true)))
    flagged = score > threshold
    if not flagged.any():
        return 0.0
    return float((flagged & y_true).sum() / flagged.sum())


def row_hashes(data):
    hashes = np.empty(len(data), dtype="S20")
    for start in range(0, len(data), _CHUNK):
        chunk = np.ascontiguousarray(data[start:start + _CHUNK])
        for i, row in enumerate(chunk):
            hashes[start + i] = hashlib.sha1(row.tobytes()).digest()
    return hashes


def score_detector(table_path, hashes):
    table = np.load(table_path)
    lookup, conflicting = {}, set()
    for k, s, p in zip(table["row_sha1"].tolist(), table["score"].tolist(), table["is_poisoned"].tolist()):
        if k in lookup and lookup[k] != (s, int(p)):
            conflicting.add(k)
        lookup.setdefault(k, (s, int(p)))

    rows = hashes.tolist()
    missing = sum(1 for h in rows if h not in lookup)
    ambiguous = sum(1 for h in rows if h in conflicting)
    result = {"n_train": len(rows), "scores_file": Path(table_path).name}
    if missing or ambiguous:
        result.update(status="unverifiable",
                      reason=f"{missing} training rows not scored by the detector, "
                             f"{ambiguous} rows with conflicting duplicate entries")
        return result

    s = np.array([lookup[h][0] for h in rows], dtype=np.float64)
    y = np.array([lookup[h][1] for h in rows], dtype=np.int64)
    n_poisoned = int(y.sum())
    result.update(n_poisoned=n_poisoned, poison_rate=n_poisoned / len(rows))
    if n_poisoned in (0, len(rows)):
        result.update(status="undefined", reason="training set has a single class; ROC-AUC/P@N undefined")
        return result
    result.update(status="ok", roc_auc=roc_auc(y, s), precision_at_n=precision_at_n(y, s))
    return result


def evaluate(input_dir):
    """Return the evaluation_results.json payload for input_dir."""
    input_dir = Path(input_dir)
    tables = sorted(input_dir.glob(SCORES_GLOB))
    data_path = input_dir / "data.npy"
    if not data_path.exists() or not tables:
        missing = [n for n, ok in (("data.npy", data_path.exists()), (SCORES_GLOB, bool(tables))) if not ok]
        return {"success": False, "skipped": False, "metrics": {},
                "error": f"missing inputs: {', '.join(missing)}"}

    hashes = row_hashes(np.load(data_path, mmap_mode="r"))
    metrics, details = {}, {}
    for path in tables:
        name = path.name[len("detection_scores."):-len(".npz")]
        try:
            d = score_detector(path, hashes)
        except (KeyError, ValueError, OSError) as exc:
            d = {"status": "error", "reason": f"unreadable {path.name}: {exc}"}
        details[name] = d
        if d["status"] == "ok":
            metrics[f"{name}_train_roc_auc"] = d["roc_auc"]
            metrics[f"{name}_train_precision_at_n"] = d["precision_at_n"]
    if any(d["status"] == "error" for d in details.values()):
        return {"success": False, "skipped": False, "metrics": {}, "details": details,
                "error": "unreadable detector score table"}
    if not metrics:
        return {"success": True, "skipped": True, "metrics": {}, "details": details,
                "skip_reason": "no detector with exact training-set metrics"}
    return {"success": True, "skipped": False, "metrics": metrics, "details": details}


def main():
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    payload = evaluate(workspace / "input")
    payload.update(evaluator="sanitization", timestamp=datetime.utcnow().isoformat() + "Z")
    (output_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload.get(k) for k in ("success", "skipped", "metrics", "details", "error")}, indent=2))


if __name__ == "__main__":
    main()
