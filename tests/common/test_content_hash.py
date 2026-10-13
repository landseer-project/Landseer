"""Tests for src.common.content_hash.dataset_content_hash."""

import json
import os
from pathlib import Path

from src.common.content_hash import HASH_SIDECAR, dataset_content_hash


def _make_dataset(d: Path, data: bytes = b"\x00" * 100, labels: bytes = b"\x01" * 10) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.npy").write_bytes(data)
    (d / "labels.npy").write_bytes(labels)
    (d / "test_data.npy").write_bytes(data[:50])
    (d / "test_labels.npy").write_bytes(labels[:5])
    return d


def test_same_content_different_dirs_and_mtimes_hash_equal(tmp_path):
    a = _make_dataset(tmp_path / "host_a" / "cifar")
    b = _make_dataset(tmp_path / "elsewhere" / "copy")
    os.utime(b / "data.npy", ns=(1_000_000_000, 1_000_000_000))
    h = dataset_content_hash(a)
    assert h is not None and len(h) == 64
    assert dataset_content_hash(b) == h


def test_different_content_differs(tmp_path):
    a = _make_dataset(tmp_path / "a")
    b = _make_dataset(tmp_path / "b", data=b"\x09" * 100)
    assert dataset_content_hash(a) != dataset_content_hash(b)


def test_config_model_and_other_files_ignored(tmp_path):
    a = _make_dataset(tmp_path / "a")
    h = dataset_content_hash(a)
    (a / "config_model.py").write_text("class Net: pass\n")
    (a / "notes.txt").write_text("hello")
    assert dataset_content_hash(a) == h


def test_sidecar_written_and_refreshed_on_change(tmp_path):
    a = _make_dataset(tmp_path / "a")
    h1 = dataset_content_hash(a)
    sidecar = json.loads((a / HASH_SIDECAR).read_text())
    assert set(sidecar) == {"data.npy", "labels.npy", "test_data.npy", "test_labels.npy"}

    (a / "data.npy").write_bytes(b"\x07" * 101)
    h2 = dataset_content_hash(a)
    assert h2 != h1
    refreshed = json.loads((a / HASH_SIDECAR).read_text())
    assert refreshed["data.npy"]["size"] == 101
    assert refreshed["data.npy"]["sha256"] != sidecar["data.npy"]["sha256"]

    # Sidecar is not part of the hash.
    assert dataset_content_hash(a) == h2


def test_sidecar_is_trusted_only_when_size_and_mtime_match(tmp_path):
    a = _make_dataset(tmp_path / "a")
    h = dataset_content_hash(a)
    sidecar = json.loads((a / HASH_SIDECAR).read_text())
    sidecar["data.npy"]["sha256"] = "0" * 64
    (a / HASH_SIDECAR).write_text(json.dumps(sidecar))
    # Same size/mtime -> the (poisoned) cached digest is reused.
    assert dataset_content_hash(a) != h
    # Touching the file invalidates the cached digest.
    st = (a / "data.npy").stat()
    os.utime(a / "data.npy", ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    assert dataset_content_hash(a) == h


def test_none_for_missing_or_empty_dir(tmp_path):
    assert dataset_content_hash(None) is None
    assert dataset_content_hash(tmp_path / "nope") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert dataset_content_hash(empty) is None
    (empty / "config_model.py").write_text("x")
    assert dataset_content_hash(empty) is None
