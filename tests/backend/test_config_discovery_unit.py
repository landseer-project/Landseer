"""
Unit tests for the config discovery module.

Tests cover:
- discover_configs: empty when pipeline_dir doesn't exist
- discover_configs: finds yaml files in the directory
- discover_configs: config_id derived from filename stem
- discover_configs: returns sorted by filename
- discover_configs: ignores non-yaml files
- discover_configs: resolves config_path to absolute
- compute_config_hash: returns non-empty string for existing file
- compute_config_hash: returns empty string for missing file
- compute_config_hash: same content yields same hash
- compute_config_hash: different content yields different hash
- compute_config_hash: includes attack config in hash when provided
- sync_configs_to_db: creates new configs
- sync_configs_to_db: updates existing configs
- sync_configs_to_db: handles empty directory
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.backend.config_discovery import (
    discover_configs,
    compute_config_hash,
    sync_configs_to_db,
    get_all_configs,
    get_config_by_id,
    _resolve_dir,
    _hash_file,
)


# ============================================================================
# discover_configs
# ============================================================================


class TestDiscoverConfigs:
    """Tests for discover_configs()."""

    def test_empty_when_dir_missing(self, tmp_path):
        result = discover_configs(
            pipeline_dir=str(tmp_path / "nonexistent"),
            attack_dir=str(tmp_path / "attacks"),
        )
        assert result == []

    def test_empty_when_dir_has_no_yaml(self, tmp_path):
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "readme.txt").write_text("not a yaml")
        result = discover_configs(
            pipeline_dir=str(tmp_path / "pipeline"),
            attack_dir=str(tmp_path / "attacks"),
        )
        assert result == []

    def test_finds_yaml_files(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "trades.yaml").write_text("dataset:\n  name: cifar10\n")
        (pdir / "mini.yaml").write_text("dataset:\n  name: mnist\n")

        result = discover_configs(
            pipeline_dir=str(pdir),
            attack_dir=str(tmp_path / "attacks"),
        )
        assert len(result) == 2

    def test_config_id_from_stem(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "my_pipeline.yaml").write_text("x: 1\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert result[0]["id"] == "config_my_pipeline"
        assert result[0]["name"] == "my_pipeline"

    def test_configs_sorted_by_filename(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "c_third.yaml").write_text("x: 3\n")
        (pdir / "a_first.yaml").write_text("x: 1\n")
        (pdir / "b_second.yaml").write_text("x: 2\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        names = [c["name"] for c in result]
        assert names == ["a_first", "b_second", "c_third"]

    def test_ignores_non_yaml_files(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "good.yaml").write_text("x: 1\n")
        (pdir / "bad.json").write_text('{"x": 1}')
        (pdir / "bad.txt").write_text("x: 1")
        (pdir / "bad.yml").write_text("x: 1\n")  # .yml not matched, only .yaml

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert len(result) == 1
        assert result[0]["name"] == "good"

    def test_config_path_is_absolute(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "test.yaml").write_text("x: 1\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert Path(result[0]["config_path"]).is_absolute()

    def test_config_has_description(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "test.yaml").write_text("x: 1\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert "Discovered from" in result[0]["description"]

    def test_config_has_hash(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "test.yaml").write_text("x: 1\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert result[0]["config_hash"]
        assert len(result[0]["config_hash"]) == 64  # SHA-256 hex digest

    def test_attack_config_path_is_none(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "test.yaml").write_text("x: 1\n")

        result = discover_configs(pipeline_dir=str(pdir), attack_dir=str(tmp_path))
        assert result[0]["attack_config_path"] is None

    def test_discovers_real_configs_directory(self):
        """Verify that the actual configs/pipeline/ directory is discoverable."""
        repo_root = Path(__file__).parents[2]
        pdir = repo_root / "configs" / "pipeline"
        if not pdir.exists():
            pytest.skip("configs/pipeline/ not found in repo")

        result = discover_configs(
            pipeline_dir=str(pdir),
            attack_dir=str(repo_root / "configs" / "attack"),
        )
        assert len(result) > 0
        names = [c["name"] for c in result]
        assert "trades" in names or "mini" in names

    def test_relative_path_resolves_from_repo_when_cwd_differs(self, tmp_path):
        """
        Regression test for UI showing 0 configs when backend runs from another cwd.
        """
        repo_root = Path(__file__).parents[2]
        pdir = repo_root / "configs" / "pipeline"
        if not pdir.exists():
            pytest.skip("configs/pipeline/ not found in repo")

        old_cwd = Path.cwd()
        try:
            # Move to an unrelated directory so relative discovery would fail
            os.chdir(tmp_path)
            result = discover_configs(pipeline_dir="configs/pipeline")
        finally:
            os.chdir(old_cwd)

        assert len(result) > 0
        names = [c["name"] for c in result]
        assert "trades" in names or "mini" in names


# ============================================================================
# compute_config_hash
# ============================================================================


class TestComputeConfigHash:
    """Tests for compute_config_hash()."""

    def test_returns_nonempty_for_existing_file(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("dataset:\n  name: cifar10\n")
        h = compute_config_hash(str(f))
        assert h
        assert len(h) == 64

    def test_returns_empty_for_missing_file(self):
        h = compute_config_hash("/nonexistent/path/config.yaml")
        assert h == ""

    def test_same_content_same_hash(self, tmp_path):
        content = "dataset:\n  name: cifar10\n"
        f1 = tmp_path / "a.yaml"
        f2 = tmp_path / "b.yaml"
        f1.write_text(content)
        f2.write_text(content)
        assert compute_config_hash(str(f1)) == compute_config_hash(str(f2))

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f2 = tmp_path / "b.yaml"
        f1.write_text("dataset:\n  name: cifar10\n")
        f2.write_text("dataset:\n  name: mnist\n")
        assert compute_config_hash(str(f1)) != compute_config_hash(str(f2))

    def test_attack_config_changes_hash(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("x: 1\n")
        attack = tmp_path / "attack.yaml"
        attack.write_text("attack: pgd\n")

        h_without = compute_config_hash(str(f))
        h_with = compute_config_hash(str(f), attack_config_path=str(attack))
        assert h_without != h_with

    def test_missing_attack_config_ignored(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("x: 1\n")
        h_without = compute_config_hash(str(f))
        h_with = compute_config_hash(str(f), attack_config_path="/nonexistent/attack.yaml")
        assert h_without == h_with


# ============================================================================
# _hash_file
# ============================================================================


class TestHashFile:
    """Tests for _hash_file()."""

    def test_returns_hex_digest(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = _hash_file(f)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _hash_file(f) == _hash_file(f)


# ============================================================================
# _resolve_dir
# ============================================================================


class TestResolveDir:
    """Tests for _resolve_dir()."""

    def test_absolute_path_unchanged(self, tmp_path):
        resolved = _resolve_dir(str(tmp_path))
        assert resolved == tmp_path.resolve()

    def test_relative_prefers_existing_cwd(self, tmp_path):
        (tmp_path / "configs" / "pipeline").mkdir(parents=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            resolved = _resolve_dir("configs/pipeline")
        finally:
            os.chdir(old_cwd)
        assert resolved == (tmp_path / "configs" / "pipeline").resolve()


# ============================================================================
# sync_configs_to_db
# ============================================================================


class TestSyncConfigsToDb:
    """Tests for sync_configs_to_db() using mocked DB layer."""

    def test_creates_new_configs(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "mini.yaml").write_text("x: 1\n")
        (pdir / "trades.yaml").write_text("x: 2\n")

        created = []
        updated = []

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_by_id(self, config_id):
                return None
            def create(self, cfg):
                created.append(cfg)
                return MagicMock(id=cfg["id"])

        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = sync_configs_to_db(
                pipeline_dir=str(pdir),
                attack_dir=str(tmp_path / "attacks"),
            )

        assert len(created) == 2
        assert created[0]["id"] == "config_mini"
        assert created[1]["id"] == "config_trades"

    def test_updates_existing_configs(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()
        (pdir / "mini.yaml").write_text("x: 1\n")

        updated_data = []

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_by_id(self, config_id):
                return MagicMock(id=config_id)
            def update(self, config_id, updates):
                updated_data.append((config_id, updates))
                return MagicMock(id=config_id)

        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            sync_configs_to_db(
                pipeline_dir=str(pdir),
                attack_dir=str(tmp_path / "attacks"),
            )

        assert len(updated_data) == 1
        config_id, updates = updated_data[0]
        assert config_id == "config_mini"
        assert "config_path" in updates
        assert "config_hash" in updates

    def test_empty_directory(self, tmp_path):
        pdir = tmp_path / "pipeline"
        pdir.mkdir()

        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = sync_configs_to_db(
                pipeline_dir=str(pdir),
                attack_dir=str(tmp_path / "attacks"),
            )

        assert result == []

    def test_missing_directory(self, tmp_path):
        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = sync_configs_to_db(
                pipeline_dir=str(tmp_path / "nonexistent"),
                attack_dir=str(tmp_path / "attacks"),
            )

        assert result == []


# ============================================================================
# get_all_configs
# ============================================================================


class TestGetAllConfigs:
    """Tests for get_all_configs()."""

    def test_returns_configs_from_repo(self):
        fake_configs = [MagicMock(id="config_a"), MagicMock(id="config_b")]

        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_all(self):
                return fake_configs

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = get_all_configs()

        assert len(result) == 2
        assert result[0].id == "config_a"
        assert result[1].id == "config_b"

    def test_returns_empty_list(self):
        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_all(self):
                return []

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = get_all_configs()

        assert result == []

    def test_propagates_db_error(self):
        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_all(self):
                raise RuntimeError("DB unavailable")

        with pytest.raises(RuntimeError, match="DB unavailable"):
            with patch("src.backend.config_discovery.session_scope", _FakeSession), \
                 patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
                get_all_configs()


# ============================================================================
# get_config_by_id
# ============================================================================


class TestGetConfigById:
    """Tests for get_config_by_id()."""

    def test_returns_config_when_found(self):
        fake_config = MagicMock(id="config_trades", name="trades")

        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_by_id(self, config_id):
                return fake_config if config_id == "config_trades" else None

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = get_config_by_id("config_trades")

        assert result is not None
        assert result.id == "config_trades"

    def test_returns_none_when_not_found(self):
        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_by_id(self, config_id):
                return None

        with patch("src.backend.config_discovery.session_scope", _FakeSession), \
             patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
            result = get_config_by_id("config_nonexistent")

        assert result is None

    def test_propagates_db_error(self):
        class _FakeSession:
            def __enter__(self):
                return object()
            def __exit__(self, *args):
                pass

        class _FakeRepo:
            def __init__(self, _session):
                pass
            def get_by_id(self, config_id):
                raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            with patch("src.backend.config_discovery.session_scope", _FakeSession), \
                 patch("src.backend.config_discovery.PipelineConfigRepository", _FakeRepo):
                get_config_by_id("config_any")
