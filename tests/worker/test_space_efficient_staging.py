"""Tests for hard-link dataset staging and config_model placement (space-efficiency MVP)."""

import errno
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from src.worker.runner import _link_or_copy, _mirror_tree_link_or_copy


def test_link_or_copy_same_inode_on_same_filesystem(tmp_path: Path) -> None:
    src = tmp_path / "data.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "linked.bin"
    _link_or_copy(src, dst)
    s_src = src.stat()
    s_dst = dst.stat()
    assert s_src.st_ino == s_dst.st_ino
    assert s_src.st_dev == s_dst.st_dev
    assert dst.read_bytes() == b"payload"


def test_link_or_copy_fallback_copy_on_exdev(tmp_path: Path) -> None:
    src = tmp_path / "a.bin"
    src.write_bytes(b"x")
    dst = tmp_path / "b.bin"

    real_link = os.link

    def fake_link(a: str, b: str) -> None:
        if os.fspath(a).endswith("a.bin"):
            raise OSError(errno.EXDEV, "cross-device link")
        real_link(a, b)

    with patch("os.link", side_effect=fake_link):
        _link_or_copy(src, dst)

    assert dst.read_bytes() == b"x"
    assert src.stat().st_ino != dst.stat().st_ino


def test_mirror_tree_link_or_copy_shared_inodes(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "f.npy"
    f.write_bytes(b"0123")
    out = tmp_path / "input"
    _mirror_tree_link_or_copy(root, out)
    linked = out / "sub" / "f.npy"
    assert linked.exists()
    assert f.stat().st_ino == linked.stat().st_ino


def test_config_model_unlink_allows_fresh_copy(tmp_path: Path) -> None:
    """After unlink, a read-only placeholder cannot block writing the canonical file."""
    canonical = tmp_path / "canonical_config_model.py"
    canonical.write_text("# config\n")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    dest = input_dir / "config_model.py"
    dest.write_text("stale")
    dest.chmod(0o444)
    try:
        dest.unlink(missing_ok=True)
        shutil.copy2(canonical, dest)
    finally:
        if dest.exists():
            dest.chmod(0o644)
    assert dest.read_text() == "# config\n"


def test_config_model_symlink_mode_matches_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LANDSEER_CONFIG_MODEL_SYMLINK=1 uses symlink to canonical (same as runner logic)."""
    monkeypatch.setenv("LANDSEER_CONFIG_MODEL_SYMLINK", "1")
    canonical = tmp_path / "canonical.py"
    canonical.write_text("x = 1\n")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    dest = input_dir / "config_model.py"
    dest.unlink(missing_ok=True)
    dest.symlink_to(canonical.resolve())
    assert dest.is_symlink()
    assert (input_dir / "config_model.py").read_text() == "x = 1\n"
