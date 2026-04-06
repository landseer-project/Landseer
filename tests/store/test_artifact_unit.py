"""
Artifact manager unit tests.

Tests cover:
- ArtifactInfo.to_dict: correct keys and values
- ArtifactInfo: local_path rendered as string in dict
- ArtifactManager.compute_cache_key: deterministic for same inputs
- ArtifactManager.compute_cache_key: different inputs → different key
- ArtifactManager.compute_cache_key: parent_hashes are sorted (order-independent)
- ArtifactManager.get_local_path: returns correct path under cache dir
- ArtifactManager.local_exists: False when dir absent, False when .success missing
- ArtifactManager.local_exists: True when dir and .success present
- ArtifactManager.get_local_artifact: None when not exists, Path when exists
- ArtifactManager.store_local: creates directory, copies files, writes .success
- ArtifactManager.store_local: returns ArtifactInfo with correct fields
- ArtifactManager: initialized without MinIO when use_minio=False
- ArtifactManager: creates local_cache_dir on init
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.store.artifact_manager import ArtifactInfo, ArtifactManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary cache directory."""
    d = tmp_path / "artifact_cache"
    return d  # let ArtifactManager create it


@pytest.fixture
def manager(cache_dir: Path) -> ArtifactManager:
    """ArtifactManager with MinIO disabled."""
    return ArtifactManager(local_cache_dir=cache_dir, use_minio=False)


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """A source directory with a single output file."""
    src = tmp_path / "task_output"
    src.mkdir()
    (src / "model.pth").write_bytes(b"\x00" * 128)
    (src / "metrics.json").write_text('{"acc": 0.95}')
    return src


# ============================================================================
# Tests: ArtifactInfo.to_dict
# ============================================================================


class TestArtifactInfoToDict:
    """Tests for ArtifactInfo.to_dict serialization."""

    def test_contains_required_keys(self):
        info = ArtifactInfo(
            cache_key="abc123",
            task_id="task_1",
            tool_name="clean-eval",
            storage_type="local",
        )
        d = info.to_dict()
        for key in ["cache_key", "task_id", "tool_name", "storage_type",
                    "local_path", "minio_key", "size_bytes", "created_at",
                    "parent_hashes", "metadata"]:
            assert key in d, f"Missing key: {key}"

    def test_values_match_constructor(self):
        info = ArtifactInfo(
            cache_key="key-42",
            task_id="t99",
            tool_name="my-tool",
            storage_type="both",
            size_bytes=1024,
            parent_hashes=["hash1", "hash2"],
            metadata={"version": 2}
        )
        d = info.to_dict()
        assert d["cache_key"] == "key-42"
        assert d["task_id"] == "t99"
        assert d["size_bytes"] == 1024
        assert d["parent_hashes"] == ["hash1", "hash2"]
        assert d["metadata"] == {"version": 2}

    def test_local_path_rendered_as_string(self, tmp_path):
        info = ArtifactInfo(
            cache_key="k",
            task_id="t",
            tool_name="tool",
            storage_type="local",
            local_path=tmp_path / "output"
        )
        d = info.to_dict()
        assert isinstance(d["local_path"], str)
        assert str(tmp_path / "output") in d["local_path"]

    def test_none_local_path_serialized_as_none(self):
        info = ArtifactInfo(
            cache_key="k",
            task_id="t",
            tool_name="tool",
            storage_type="minio",
            local_path=None
        )
        d = info.to_dict()
        assert d["local_path"] is None

    def test_created_at_is_iso_string(self):
        from datetime import datetime
        info = ArtifactInfo(cache_key="k", task_id="t", tool_name="n", storage_type="local")
        # Should be parseable as ISO datetime
        datetime.fromisoformat(info.to_dict()["created_at"])


# ============================================================================
# Tests: ArtifactManager initialization
# ============================================================================


class TestArtifactManagerInit:
    """Tests for ArtifactManager construction."""

    def test_creates_cache_dir(self, tmp_path):
        cache = tmp_path / "new_cache"
        assert not cache.exists()
        ArtifactManager(local_cache_dir=cache, use_minio=False)
        assert cache.exists()

    def test_minio_none_when_disabled(self, cache_dir):
        mgr = ArtifactManager(local_cache_dir=cache_dir, use_minio=False)
        assert mgr.minio is None

    def test_use_minio_false_does_not_connect(self, cache_dir):
        # No exception raised even if MinIO is unavailable
        mgr = ArtifactManager(local_cache_dir=cache_dir, use_minio=False)
        assert mgr.use_minio is False


# ============================================================================
# Tests: ArtifactManager.compute_cache_key
# ============================================================================


class TestComputeCacheKey:
    """Tests for compute_cache_key hashing."""

    def test_deterministic_for_same_inputs(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {"lr": 0.01}, ["h1"])
        k2 = manager.compute_cache_key("t1", "tool", "img:v1", {"lr": 0.01}, ["h1"])
        assert k1 == k2

    def test_different_tool_name_different_key(self, manager):
        k1 = manager.compute_cache_key("t1", "tool-a", "img:v1", {}, [])
        k2 = manager.compute_cache_key("t1", "tool-b", "img:v1", {}, [])
        assert k1 != k2

    def test_different_image_different_key(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {}, [])
        k2 = manager.compute_cache_key("t1", "tool", "img:v2", {}, [])
        assert k1 != k2

    def test_different_config_different_key(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {"lr": 0.01}, [])
        k2 = manager.compute_cache_key("t1", "tool", "img:v1", {"lr": 0.1}, [])
        assert k1 != k2

    def test_different_parent_hashes_different_key(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {}, ["aaa"])
        k2 = manager.compute_cache_key("t1", "tool", "img:v1", {}, ["bbb"])
        assert k1 != k2

    def test_parent_hash_order_independent(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {}, ["aaa", "bbb"])
        k2 = manager.compute_cache_key("t1", "tool", "img:v1", {}, ["bbb", "aaa"])
        assert k1 == k2

    def test_empty_parents_different_from_nonempty(self, manager):
        k1 = manager.compute_cache_key("t1", "tool", "img:v1", {}, [])
        k2 = manager.compute_cache_key("t1", "tool", "img:v1", {}, ["h1"])
        assert k1 != k2

    def test_returns_hex_string(self, manager):
        key = manager.compute_cache_key("t1", "tool", "img:v1", {}, [])
        # blake2s produces a hex digest
        assert isinstance(key, str)
        assert all(c in "0123456789abcdef" for c in key)


# ============================================================================
# Tests: ArtifactManager.get_local_path
# ============================================================================


class TestGetLocalPath:
    """Tests for get_local_path path construction."""

    def test_path_under_cache_dir(self, manager, cache_dir):
        path = manager.get_local_path("my-cache-key")
        assert path.parent == cache_dir

    def test_path_name_is_cache_key(self, manager):
        path = manager.get_local_path("abc123")
        assert path.name == "abc123"


# ============================================================================
# Tests: ArtifactManager.local_exists
# ============================================================================


class TestLocalExists:
    """Tests for local_exists."""

    def test_false_when_dir_absent(self, manager):
        assert manager.local_exists("nonexistent-key") is False

    def test_false_when_dir_exists_but_no_success_marker(self, manager, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "missing-success").mkdir()
        assert manager.local_exists("missing-success") is False

    def test_true_when_dir_and_success_marker_present(self, manager, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = cache_dir / "ready-key"
        artifact_dir.mkdir()
        (artifact_dir / ".success").write_text("ok")

        assert manager.local_exists("ready-key") is True


# ============================================================================
# Tests: ArtifactManager.get_local_artifact
# ============================================================================


class TestGetLocalArtifact:
    """Tests for get_local_artifact."""

    def test_returns_none_when_not_cached(self, manager):
        result = manager.get_local_artifact("uncached-key")
        assert result is None

    def test_returns_output_path_when_cached(self, manager, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = cache_dir / "cached-key"
        output_dir = artifact_dir / "output"
        output_dir.mkdir(parents=True)
        (artifact_dir / ".success").write_text("ok")

        result = manager.get_local_artifact("cached-key")
        assert result == output_dir


# ============================================================================
# Tests: ArtifactManager.store_local
# ============================================================================


class TestStoreLocal:
    """Tests for store_local."""

    def test_creates_output_directory(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manager.store_local(
            cache_key="key-1",
            source_dir=source_dir,
            task_id="t1",
            tool_name="tool-a"
        )
        assert (cache_dir / "key-1" / "output").exists()

    def test_copies_source_files(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manager.store_local("key-1", source_dir, "t1", "tool")
        output = cache_dir / "key-1" / "output"
        assert (output / "model.pth").exists()
        assert (output / "metrics.json").exists()

    def test_writes_success_marker(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manager.store_local("key-2", source_dir, "t1", "tool")
        assert (cache_dir / "key-2" / ".success").exists()

    def test_returns_artifact_info(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = manager.store_local("key-3", source_dir, "t1", "tool-b",
                                   parent_hashes=["ph1"])
        assert isinstance(info, ArtifactInfo)
        assert info.cache_key == "key-3"
        assert info.task_id == "t1"
        assert info.tool_name == "tool-b"

    def test_local_exists_after_store(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manager.store_local("key-4", source_dir, "t1", "tool")
        assert manager.local_exists("key-4") is True

    def test_size_bytes_computed(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = manager.store_local("key-5", source_dir, "t1", "tool")
        assert info.size_bytes > 0

    def test_parent_hashes_stored_in_info(self, manager, source_dir, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = manager.store_local("key-6", source_dir, "t1", "tool",
                                   parent_hashes=["h1", "h2"])
        assert info.parent_hashes == ["h1", "h2"]

    def test_nonexistent_source_creates_empty_output(self, manager, tmp_path, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        nonexistent = tmp_path / "no_such_dir"
        # Should not raise; creates an empty output dir instead
        info = manager.store_local("key-7", nonexistent, "t1", "tool")
        assert (cache_dir / "key-7" / "output").exists()

    @pytest.mark.parametrize("key", ["abc-123", "sha256-deadbeef", "key_with_underscores"])
    def test_various_cache_keys(self, manager, source_dir, cache_dir, key):
        cache_dir.mkdir(parents=True, exist_ok=True)
        manager.store_local(key, source_dir, "t1", "tool")
        assert manager.local_exists(key) is True
