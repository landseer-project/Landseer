"""
Worker test fixtures.

Worker._init_components requires a reachable MinIO whenever caching is on.
Unit tests have no MinIO server, so the two-level cache is replaced with a
mock whose MinIO client reports itself available. Tests that exercise the
cache wiring patch ``src.worker.cli.TwoLevelCache`` themselves; their patch
is applied inside this one and takes precedence.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def isolated_cache_dir(tmp_path, monkeypatch):
    """Workers built without cache_dir use a per-test dir, never /data/landseer/cache."""
    cache = tmp_path / "landseer-cache"
    monkeypatch.setenv("LANDSEER_CACHE_DIR", str(cache))
    return cache


@pytest.fixture(autouse=True)
def fake_two_level_cache():
    with patch("src.worker.cli.TwoLevelCache") as mock_cls:
        cache = mock_cls.return_value
        cache.manager.minio = MagicMock(is_available=True)
        cache.get.return_value = None
        yield cache
