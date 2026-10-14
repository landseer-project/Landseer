"""Tests for the cache clone helpers in src.common.fsutil."""

import errno
import os
from pathlib import Path

import pytest

from src.common import fsutil
from src.common.fsutil import clone_file, clone_tree, make_tree_read_only, same_filesystem


def _src_tree(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "model.pt").write_bytes(b"\x01" * 4096)
    (root / "sub" / "metrics.json").write_text('{"acc": 0.9}')
    return root


def _files(root: Path):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_clone_tree_copies_content_with_cheapest_method(tmp_path):
    src = _src_tree(tmp_path / "src")
    counts = clone_tree(src, tmp_path / "dst")
    assert _files(tmp_path / "dst") == _files(src)
    assert sum(counts.values()) == 2
    assert set(counts) <= {"reflink", "hardlink", "copy"}


def test_hardlink_used_when_reflink_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil, "_reflink", lambda s, d: False)
    src = tmp_path / "a"
    src.write_bytes(b"x" * 100)
    assert clone_file(src, tmp_path / "b") == "hardlink"
    assert os.stat(src).st_ino == os.stat(tmp_path / "b").st_ino


def test_copy_fallback_on_cross_device(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil, "_reflink", lambda s, d: False)

    def _exdev(s, d):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(fsutil.os, "link", _exdev)
    src = tmp_path / "a"
    src.write_bytes(b"y" * 100)
    assert clone_file(src, tmp_path / "b") == "copy"
    assert (tmp_path / "b").read_bytes() == b"y" * 100
    assert os.stat(src).st_ino != os.stat(tmp_path / "b").st_ino


def test_unexpected_link_error_is_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(fsutil, "_reflink", lambda s, d: False)

    def _enospc(s, d):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(fsutil.os, "link", _enospc)
    src = tmp_path / "a"
    src.write_bytes(b"z")
    with pytest.raises(OSError):
        clone_file(src, tmp_path / "b")


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_hardlinked_workspace_copy_cannot_modify_cache_entry(tmp_path, monkeypatch):
    """A read-only cache entry stays intact even when the workspace shares its inodes."""
    monkeypatch.setattr(fsutil, "_reflink", lambda s, d: False)
    cache_out = _src_tree(tmp_path / "cache" / "key" / "output")
    make_tree_read_only(cache_out)
    workspace_out = tmp_path / "ws" / "task_1" / "output"
    workspace_out.parent.mkdir(parents=True)
    assert clone_tree(cache_out, workspace_out) == {"hardlink": 2}

    with pytest.raises(PermissionError):
        with open(workspace_out / "model.pt", "r+b") as fh:
            fh.write(b"corrupt")
    assert (cache_out / "model.pt").read_bytes() == b"\x01" * 4096

    # Deleting the workspace copy (pruning, re-runs) still works and leaves the cache.
    import shutil
    shutil.rmtree(workspace_out)
    assert (cache_out / "model.pt").exists()


@pytest.fixture
def reflink_dir(tmp_path):
    """tmp_path, or a fresh dir under LANDSEER_REFLINK_TEST_DIR (tmp_path is often tmpfs)."""
    base = os.environ.get("LANDSEER_REFLINK_TEST_DIR")
    if not base:
        yield tmp_path
        return
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix=".reflink-test-", dir=base))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reflinked_copy_is_independent_when_supported(reflink_dir):
    src = reflink_dir / "a"
    src.write_bytes(b"\x05" * 8192)
    dst = reflink_dir / "b"
    if clone_file(src, dst) != "reflink":
        pytest.skip(
            f"no reflink support at {reflink_dir}; set LANDSEER_REFLINK_TEST_DIR "
            "to a dir on btrfs or reflink-enabled xfs (e.g. under /data) to run it"
        )
    with open(dst, "r+b") as fh:
        fh.write(b"changed")
    assert src.read_bytes() == b"\x05" * 8192


def test_make_tree_read_only_keeps_dirs_writable(tmp_path):
    root = _src_tree(tmp_path / "t")
    make_tree_read_only(root)
    assert not os.access(root / "model.pt", os.W_OK) or os.geteuid() == 0
    assert os.access(root / "sub", os.W_OK)


def test_same_filesystem(tmp_path):
    assert same_filesystem(tmp_path, tmp_path / "not" / "created" / "yet")
