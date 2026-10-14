"""
Size-capped LRU for the local artifact cache (src/store/local_lru.py).

Covers:
- eviction order follows .last_used across two instances sharing a directory
- eviction stops at the low-water mark (80% of the cap)
- locked, recently used, protected and vetoed entries are kept
- datasets/, .locks and .tmp-* are never counted or evicted; stale .tmp-* is cleaned
- only one process evicts at a time
- the default cap
"""

import json
import os
import time
from pathlib import Path

import pytest

from src.store import local_lru
from src.store.local_lru import GB, LocalLRU, default_max_bytes

MB = 1024 * 1024


def make_entry(cache: Path, key: str, size: int, last_used: float) -> Path:
    d = cache / key
    (d / "output").mkdir(parents=True)
    (d / "output" / "blob.bin").write_bytes(b"\0" * size)
    (d / "manifest.json").write_text(json.dumps({"cache_key": key, "size_bytes": size}))
    (d / ".success").touch()
    (d / ".last_used").touch()
    os.utime(d / ".last_used", (last_used, last_used))
    return d


@pytest.fixture
def cache(tmp_path) -> Path:
    c = tmp_path / "cache"
    c.mkdir()
    return c


def lru(cache: Path, cap: int, **kw) -> LocalLRU:
    kw.setdefault("grace_seconds", 0)
    return LocalLRU(cache, max_bytes=cap, **kw)


def test_evicts_least_recently_used_across_instances(cache):
    now = time.time()
    for i, key in enumerate(["a", "b", "c", "d"]):
        make_entry(cache, key, 1 * MB, now - 1000 + i)  # a oldest ... d newest
    worker0 = lru(cache, 4 * MB)
    worker1 = lru(cache, 4 * MB)

    # Worker 0 uses "a", so it becomes the most recent for everyone.
    worker0.touch("a")
    evicted = worker1.ensure_space(incoming_bytes=1 * MB)

    # Need total <= 0.8 * 4 MB - 1 MB = 2.2 MB -> drop the two oldest: b, c.
    assert evicted == ["b", "c"]
    assert sorted(worker0.entries()) == ["a", "d"]


def test_no_eviction_under_cap(cache):
    make_entry(cache, "a", 1 * MB, time.time() - 100)
    assert lru(cache, 4 * MB).ensure_space(1 * MB) == []
    assert lru(cache, 4 * MB).entries() == ["a"]


def test_locked_entry_is_skipped(cache):
    now = time.time()
    make_entry(cache, "old", 2 * MB, now - 1000)
    make_entry(cache, "newer", 2 * MB, now - 500)
    holder = lru(cache, 3 * MB)
    evictor = lru(cache, 3 * MB)
    with holder.entry_lock("old"):
        evicted = evictor.ensure_space()
    assert evicted == ["newer"]
    assert (cache / "old").exists()


def test_recently_used_entries_are_kept(cache):
    now = time.time()
    make_entry(cache, "stale", 2 * MB, now - 3600)
    make_entry(cache, "fresh", 2 * MB, now - 5)
    # The cap would require evicting both; the fresh one is within the grace period.
    evicted = lru(cache, 1 * MB, grace_seconds=600).ensure_space()
    assert evicted == ["stale"]
    assert (cache / "fresh").exists()


def test_protect_and_veto(cache):
    now = time.time()
    make_entry(cache, "a", 2 * MB, now - 300)
    make_entry(cache, "b", 2 * MB, now - 200)
    make_entry(cache, "c", 2 * MB, now - 100)
    evicted = lru(cache, 3 * MB).ensure_space(protect="a", can_evict=lambda k: k != "b")
    assert evicted == ["c"]
    assert (cache / "a").exists() and (cache / "b").exists()


def test_veto_exception_keeps_entry(cache):
    make_entry(cache, "a", 2 * MB, time.time() - 300)

    def _boom(key):
        raise RuntimeError("minio down")

    assert lru(cache, 1 * MB).ensure_space(can_evict=_boom) == []
    assert (cache / "a").exists()


def test_non_entries_are_ignored(cache):
    now = time.time()
    (cache / "datasets" / "cifar10" / "clean").mkdir(parents=True)
    (cache / "datasets" / "cifar10" / "clean" / "data.npy").write_bytes(b"\0" * (5 * MB))
    (cache / ".locks").mkdir()
    staging = cache / ".tmp-a-1-abcd"
    (staging / "output").mkdir(parents=True)
    (staging / "output" / "x").write_bytes(b"\0" * MB)
    (staging / ".success").touch()
    incomplete = cache / "half"
    (incomplete / "output").mkdir(parents=True)  # no .success
    make_entry(cache, "a", 1 * MB, now - 100)

    l = lru(cache, 10 * MB)
    assert l.entries() == ["a"]
    assert l.total_size() == 1 * MB
    l.max_bytes = 1
    assert l.ensure_space() == ["a"]
    assert (cache / "datasets" / "cifar10" / "clean" / "data.npy").exists()
    assert staging.exists() and incomplete.exists()


def test_stale_staging_dirs_are_removed(cache):
    old = cache / ".tmp-x-1-dead"
    old.mkdir()
    long_ago = time.time() - 2 * 24 * 3600
    os.utime(old, (long_ago, long_ago))
    young = cache / ".tmp-y-2-live"
    young.mkdir()
    assert lru(cache, GB).remove_stale_staging() == 1
    assert not old.exists() and young.exists()


def test_only_one_evictor_at_a_time(cache):
    make_entry(cache, "a", 2 * MB, time.time() - 300)
    first = lru(cache, 1 * MB)
    second = lru(cache, 1 * MB)
    with first.entry_lock(".evict"):
        assert second.ensure_space() == []
    assert second.ensure_space() == ["a"]


def test_size_falls_back_to_walking_output_and_is_recorded(cache):
    d = make_entry(cache, "legacy", 3 * MB, time.time())
    (d / "manifest.json").write_text("{}")
    l = lru(cache, GB)
    assert l.entry_size("legacy") == 3 * MB
    assert (d / ".size").read_text() == str(3 * MB)


def test_last_used_falls_back_to_success_mtime(cache):
    d = make_entry(cache, "a", 1, time.time())
    (d / ".last_used").unlink()
    os.utime(d / ".success", (1000, 1000))
    assert lru(cache, GB).last_used("a") == 1000


def test_fits(cache):
    l = lru(cache, 10 * MB)
    assert l.fits(10 * MB) and not l.fits(10 * MB + 1)


def test_default_cap_env_and_filesystem(cache, monkeypatch):
    monkeypatch.setenv("LANDSEER_CACHE_MAX_SIZE_GB", "12.5")
    assert default_max_bytes(cache) == int(12.5 * GB)
    monkeypatch.delenv("LANDSEER_CACHE_MAX_SIZE_GB")
    st = os.statvfs(cache)
    expected = int(min(500 * GB, 0.4 * st.f_frsize * st.f_blocks))
    assert default_max_bytes(cache) == expected
