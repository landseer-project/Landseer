"""Tests for src.common.dataset_manifest."""

import json
import os
import shutil
from pathlib import Path

import pytest

from src.common import dataset_manifest as dsm


def _dataset(d: Path, fill: bytes = b"\x00") -> Path:
    (d / "img_align").mkdir(parents=True)
    (d / "data.npy").write_bytes(fill * 1000)
    (d / "labels.npy").write_bytes(fill * 10)
    (d / "img_align" / "000001.jpg").write_bytes(fill * 50)
    (d / "img_align" / "000002.jpg").write_bytes(fill * 60)
    # Not part of the dataset:
    (d / "config_model.py").write_text("NET = 1\n")
    (d / "dataset_meta.json").write_text("{}")
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_text("hidden")
    (d / "stray.log").write_text("log")
    return d


REQ_FILES = ["data.npy", "labels.npy", "missing.npy"]
REQ_DIRS = ["img_align"]


def test_selection_uses_registry_lists_and_ignore_list(tmp_path):
    d = _dataset(tmp_path / "ds")
    assert dsm.select_files(d, REQ_FILES, REQ_DIRS) == [
        "data.npy", "img_align/000001.jpg", "img_align/000002.jpg", "labels.npy",
    ]


def test_selection_without_lists_takes_every_non_ignored_file(tmp_path):
    d = _dataset(tmp_path / "ds")
    assert "stray.log" in dsm.select_files(d)
    for ignored in ("config_model.py", "dataset_meta.json", ".cache/x"):
        assert ignored not in dsm.select_files(d)


def test_content_id_independent_of_location_mtime_and_provenance(tmp_path):
    a = _dataset(tmp_path / "host_a" / "celeba")
    b = tmp_path / "elsewhere" / "copy"
    shutil.copytree(a, b)
    os.utime(b / "data.npy", ns=(10 ** 9, 10 ** 9))
    ma = dsm.build_manifest(a, REQ_FILES, REQ_DIRS, provenance={"image": "img:v1", "prep_host": "simurgh"})
    mb = dsm.build_manifest(b, REQ_FILES, REQ_DIRS, provenance={"image": "img:v2", "prep_host": "gilbreth"})
    assert ma["content_id"] == mb["content_id"]
    assert ma["provenance"]["provenance_id"] != mb["provenance"]["provenance_id"]
    assert ma["file_count"] == 4 and ma["total_bytes"] == 1120
    assert ma["schema_version"] == dsm.SCHEMA_VERSION


def test_content_id_changes_with_content(tmp_path):
    a = _dataset(tmp_path / "a")
    b = _dataset(tmp_path / "b", fill=b"\x01")
    assert dsm.build_manifest(a, REQ_FILES, REQ_DIRS)["content_id"] != \
        dsm.build_manifest(b, REQ_FILES, REQ_DIRS)["content_id"]


def test_ignored_and_unlisted_files_do_not_change_content_id(tmp_path):
    d = _dataset(tmp_path / "ds")
    before = dsm.build_manifest(d, REQ_FILES, REQ_DIRS)["content_id"]
    (d / "config_model.py").write_text("NET = 2\n")
    (d / "another.log").write_text("x")
    assert dsm.build_manifest(d, REQ_FILES, REQ_DIRS)["content_id"] == before


def test_provenance_id_ignores_host_and_timestamps():
    base = {"name": "celeba", "image": "img@sha256:abc", "params": {"a": 1}}
    one = dsm.with_provenance_id(dict(base, prep_host="h1", prepared_at="2026-01-01"))
    two = dsm.with_provenance_id(dict(base, prep_host="h2", prepared_at="2026-02-02"))
    three = dsm.with_provenance_id(dict(base, params={"a": 2}))
    assert one["provenance_id"] == two["provenance_id"] != three["provenance_id"]


def test_sidecar_reused_and_refreshed(tmp_path, monkeypatch):
    d = _dataset(tmp_path / "ds")
    dsm.build_manifest(d, REQ_FILES, REQ_DIRS)
    sidecar = json.loads((d / dsm.HASH_SIDECAR).read_text())
    assert set(sidecar) == {"data.npy", "labels.npy", "img_align/000001.jpg", "img_align/000002.jpg"}

    hashed = []
    real = dsm.file_sha256
    monkeypatch.setattr(dsm, "file_sha256", lambda p: hashed.append(Path(p).name) or real(p))
    dsm.build_manifest(d, REQ_FILES, REQ_DIRS)
    assert hashed == []  # everything came from the sidecar

    (d / "labels.npy").write_bytes(b"\x09" * 11)
    m = dsm.build_manifest(d, REQ_FILES, REQ_DIRS)
    assert hashed == ["labels.npy"]
    assert m["files"]["labels.npy"]["size"] == 11


def test_write_load_and_summarize(tmp_path):
    d = _dataset(tmp_path / "ds")
    m = dsm.build_manifest(d, REQ_FILES, REQ_DIRS, provenance={"name": "celeba"})
    dsm.write_manifest(d, m)
    loaded = dsm.load_manifest(d)
    assert loaded == json.loads(json.dumps(m))
    s = dsm.summarize(loaded)
    assert s["content_id"] == m["content_id"] and s["provenance_id"] == m["provenance"]["provenance_id"]
    # The manifest itself is not part of the dataset.
    assert dsm.build_manifest(d, REQ_FILES, REQ_DIRS)["content_id"] == m["content_id"]
    assert dsm.parse_manifest(b"not json") is None
    assert dsm.parse_manifest(json.dumps({"files": {}})) is None


def test_verify_detects_missing_size_and_hash_mismatch(tmp_path):
    src = _dataset(tmp_path / "src")
    m = dsm.build_manifest(src, REQ_FILES, REQ_DIRS)
    copy = tmp_path / "copy"
    shutil.copytree(src, copy)
    assert dsm.verify_dir(copy, m).ok

    (copy / "img_align" / "000001.jpg").unlink()
    (copy / "labels.npy").write_bytes(b"\x00" * 12)
    data = bytearray((copy / "data.npy").read_bytes())
    data[500] = 7
    (copy / "data.npy").write_bytes(bytes(data))
    res = dsm.verify_dir(copy, m)
    assert not res.ok
    assert res.missing == ["img_align/000001.jpg"]
    assert res.size_mismatch == ["labels.npy"]
    assert res.hash_mismatch == ["data.npy"]
    assert res.content_id and res.content_id != m["content_id"]


def test_unlisted_files(tmp_path):
    d = _dataset(tmp_path / "ds")
    m = dsm.build_manifest(d, REQ_FILES, REQ_DIRS)
    assert dsm.list_unlisted_files(d, m) == ["stray.log"]
