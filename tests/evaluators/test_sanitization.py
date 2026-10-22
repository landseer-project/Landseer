"""Sanitization evaluator: detector ROC-AUC / Precision@n on the training set, matched by row hash."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[2] / "containers" / "evals" / "sanitization" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("eval_sanitization", _PATH)
san = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(san)


def _pairwise_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))


def _write_table(dir_: Path, data, scores, poisoned, name="xgbod"):
    hashes = np.array([hashlib.sha1(np.ascontiguousarray(r).tobytes()).digest() for r in data], dtype="S20")
    np.savez(dir_ / f"detection_scores.{name}.npz", row_sha1=hashes,
             score=np.asarray(scores, dtype=np.float64), is_poisoned=np.asarray(poisoned, dtype=np.int8))


def _dataset(seed, n=300):
    rng = np.random.RandomState(seed)
    data = rng.rand(n, 3, 4, 4).astype(np.float32)
    poisoned = (rng.rand(n) < 0.15).astype(int)
    return rng, data, poisoned, rng.rand(n) + poisoned * 0.6


@pytest.mark.parametrize("seed", range(5))
def test_roc_auc_matches_pairwise_definition_with_ties(seed):
    rng = np.random.RandomState(seed)
    y = (rng.rand(400) < 0.2).astype(int)
    s = np.round(rng.rand(400) + y * 0.5, 1)
    assert san.roc_auc(y, s) == pytest.approx(_pairwise_auc(y, s), abs=1e-12)


def test_precision_at_n_follows_authors_get_precn():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    # 40% outliers -> 60th percentile (linear) = 0.74; flagged {0.9, 0.8}: 1 of 2 correct.
    assert san.precision_at_n(y, s) == pytest.approx(0.5)
    assert san.precision_at_n(y, np.zeros(5)) == 0.0  # all tied: nothing strictly above


def test_metrics_on_training_rows_after_removal_and_reorder(tmp_path):
    rng, data, poisoned, scores = _dataset(0)
    _write_table(tmp_path, data, scores, poisoned)
    keep = rng.permutation(np.flatnonzero(scores < 1.2))
    np.save(tmp_path / "data.npy", data[keep])

    out = san.evaluate(tmp_path)
    assert out["success"] and not out["skipped"]
    assert out["metrics"]["xgbod_train_roc_auc"] == pytest.approx(_pairwise_auc(poisoned[keep], scores[keep]), abs=1e-12)
    assert out["metrics"]["xgbod_train_precision_at_n"] == pytest.approx(san.precision_at_n(poisoned[keep], scores[keep]))
    assert out["details"]["xgbod"]["n_train"] == len(keep)
    assert out["details"]["xgbod"]["n_poisoned"] == int(poisoned[keep].sum())


def test_changed_training_row_reports_no_metrics(tmp_path):
    _, data, poisoned, scores = _dataset(1)
    _write_table(tmp_path, data, scores, poisoned)
    changed = data.copy()
    changed[3, 0, 0, 0] += 1e-3
    np.save(tmp_path / "data.npy", changed)
    out = san.evaluate(tmp_path)
    assert out["success"] and out["skipped"] and out["metrics"] == {}
    assert out["details"]["xgbod"]["status"] == "unverifiable"


def test_conflicting_duplicate_rows_report_no_metrics(tmp_path):
    _, data, poisoned, scores = _dataset(2)
    data[1] = data[0]
    poisoned[0], poisoned[1] = 0, 1
    _write_table(tmp_path, data, scores, poisoned)
    np.save(tmp_path / "data.npy", data)
    assert san.evaluate(tmp_path)["details"]["xgbod"]["status"] == "unverifiable"


def test_single_class_training_set_is_undefined(tmp_path):
    _, data, poisoned, scores = _dataset(3)
    _write_table(tmp_path, data, scores, poisoned)
    np.save(tmp_path / "data.npy", data[poisoned == 0])
    out = san.evaluate(tmp_path)
    assert out["skipped"] and out["details"]["xgbod"]["status"] == "undefined"


def test_missing_inputs_fail(tmp_path):
    np.save(tmp_path / "data.npy", np.zeros((4, 2), np.float32))
    out = san.evaluate(tmp_path)
    assert out["success"] is False and "detection_scores" in out["error"]


def test_unreadable_table_fails(tmp_path):
    _, data, poisoned, scores = _dataset(4)
    np.save(tmp_path / "data.npy", data)
    np.savez(tmp_path / "detection_scores.xgbod.npz", wrong=np.zeros(3))
    assert san.evaluate(tmp_path)["success"] is False
