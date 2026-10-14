"""
Size-capped, least-recently-used local artifact cache.

Entries live in ``<cache_dir>/<cache_key>/`` (``output/``, ``manifest.json``,
``.success``). This module only decides which entries to keep:

- every use touches ``<entry>/.last_used``, so recency is shared by all worker
  processes on the host and survives restarts;
- each entry's size is read from ``manifest.json`` (``size_bytes``) or a
  ``.size`` file, so totals never walk the output trees;
- once the total passes the cap, the least recently used entries are removed
  down to the low-water mark, under the same per-entry flock the
  ArtifactManager uses for writers. Entries that are locked, or were used
  within the grace period, are never evicted;
- ``datasets/``, ``.locks`` and ``.tmp-*`` staging dirs are not entries.

Env:
- LANDSEER_CACHE_MAX_SIZE_GB: cap (default: min(500 GB, 40% of the cache
  filesystem))
- LANDSEER_CACHE_EVICT_GRACE_SECONDS: recently used entries are kept (default 600)
"""

import errno
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

from ..common import get_logger

logger = get_logger(__name__)

LAST_USED = ".last_used"
SIZE_FILE = ".size"
SUCCESS = ".success"
MANIFEST = "manifest.json"
LOCK_DIR = ".locks"
STAGING_PREFIX = ".tmp-"
NOT_ENTRIES = {"datasets"}

GB = 1024 ** 3
DEFAULT_CAP_CEILING = 500 * GB
DEFAULT_CAP_FRACTION = 0.40
DEFAULT_GRACE_SECONDS = 600
LOW_WATER = 0.8
# Staging dirs older than this belong to crashed processes.
STALE_STAGING_SECONDS = 24 * 3600


def default_max_bytes(cache_dir: Path) -> int:
    """LANDSEER_CACHE_MAX_SIZE_GB, else the lesser of 500 GB and 40% of the filesystem."""
    raw = os.environ.get("LANDSEER_CACHE_MAX_SIZE_GB")
    if raw:
        try:
            return int(float(raw) * GB)
        except ValueError:
            logger.warning(f"Ignoring invalid LANDSEER_CACHE_MAX_SIZE_GB={raw!r}")
    try:
        st = os.statvfs(cache_dir)
        fs_bytes = st.f_frsize * st.f_blocks
    except (OSError, AttributeError):
        return DEFAULT_CAP_CEILING
    return int(min(DEFAULT_CAP_CEILING, DEFAULT_CAP_FRACTION * fs_bytes))


def _default_grace() -> float:
    try:
        return float(os.environ.get("LANDSEER_CACHE_EVICT_GRACE_SECONDS", DEFAULT_GRACE_SECONDS))
    except ValueError:
        return float(DEFAULT_GRACE_SECONDS)


class LocalLRU:
    """Recency, size accounting and eviction for one local cache directory."""

    def __init__(
        self,
        cache_dir: Path,
        max_bytes: Optional[int] = None,
        grace_seconds: Optional[float] = None,
        low_water: float = LOW_WATER,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes) if max_bytes is not None else default_max_bytes(self.cache_dir)
        self.grace_seconds = _default_grace() if grace_seconds is None else float(grace_seconds)
        self.low_water = low_water

    # ------------------------------------------------------------------
    # Locks
    # ------------------------------------------------------------------

    def lock_path(self, cache_key: str) -> Path:
        return self.cache_dir / LOCK_DIR / f"{cache_key}.lock"

    @contextmanager
    def entry_lock(self, cache_key: str, blocking: bool = True) -> Iterator[bool]:
        """flock on the entry's lock file. Yields False if non-blocking and busy."""
        if fcntl is None:
            yield True
            return
        path = self.lock_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+") as fh:
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fh.fileno(), flags)
            except OSError as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    yield False
                    return
                raise
            try:
                yield True
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Recency and size
    # ------------------------------------------------------------------

    def entry_dir(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key

    def touch(self, cache_key: str) -> None:
        """Record a use of the entry now."""
        p = self.entry_dir(cache_key) / LAST_USED
        try:
            p.touch()
            os.utime(p, None)
        except OSError:
            pass

    def last_used(self, cache_key: str) -> float:
        d = self.entry_dir(cache_key)
        for name in (LAST_USED, SUCCESS):
            try:
                return (d / name).stat().st_mtime
            except OSError:
                continue
        try:
            return d.stat().st_mtime
        except OSError:
            return 0.0

    def entry_size(self, cache_key: str) -> int:
        """Bytes in the entry's output, from metadata when available."""
        d = self.entry_dir(cache_key)
        try:
            return int((d / SIZE_FILE).read_text().strip())
        except (OSError, ValueError):
            pass
        try:
            size = json.loads((d / MANIFEST).read_text()).get("size_bytes")
            if isinstance(size, int) and size >= 0:
                return size
        except (OSError, ValueError, AttributeError):
            pass
        size = 0
        for root, _dirs, files in os.walk(d / "output"):
            for name in files:
                try:
                    size += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        self.record_size(cache_key, size)
        return size

    def record_size(self, cache_key: str, size: int) -> None:
        try:
            (self.entry_dir(cache_key) / SIZE_FILE).write_text(str(int(size)))
        except OSError:
            pass

    def fits(self, size: int) -> bool:
        """Whether an entry of this size may be kept at all."""
        return size <= self.max_bytes

    def entries(self) -> List[str]:
        """Complete entries currently on disk."""
        keys = []
        try:
            children = list(self.cache_dir.iterdir())
        except OSError:
            return keys
        for p in children:
            name = p.name
            if name.startswith(".") or name in NOT_ENTRIES:
                continue
            if p.is_dir() and (p / SUCCESS).exists():
                keys.append(name)
        return keys

    def total_size(self) -> int:
        return sum(self.entry_size(k) for k in self.entries())

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def remove_entry(self, cache_key: str) -> None:
        shutil.rmtree(self.entry_dir(cache_key), ignore_errors=True)

    def remove_stale_staging(self, max_age: float = STALE_STAGING_SECONDS) -> int:
        """Delete .tmp-* staging dirs left behind by crashed processes."""
        removed = 0
        cutoff = time.time() - max_age
        try:
            children = list(self.cache_dir.iterdir())
        except OSError:
            return 0
        for p in children:
            if not p.name.startswith(STAGING_PREFIX):
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    def ensure_space(
        self,
        incoming_bytes: int = 0,
        can_evict: Optional[Callable[[str], bool]] = None,
        protect: Optional[str] = None,
    ) -> List[str]:
        """
        Make room for ``incoming_bytes``: if the total would pass the cap,
        evict least-recently-used entries until it is under the low-water mark.

        ``can_evict(key)`` may veto a candidate (e.g. not yet in MinIO).
        ``protect`` is never evicted (the entry being written). Returns the
        evicted keys. Only one process evicts at a time; others skip.
        """
        evicted: List[str] = []
        with self.entry_lock(".evict", blocking=False) as got:
            if not got:
                return evicted
            self.remove_stale_staging()

            sizes: Dict[str, int] = {k: self.entry_size(k) for k in self.entries()}
            total = sum(sizes.values())
            if total + incoming_bytes <= self.max_bytes:
                return evicted

            target = max(0, int(self.max_bytes * self.low_water) - incoming_bytes)
            now = time.time()
            for key in sorted(sizes, key=self.last_used):
                if total <= target:
                    break
                if key == protect:
                    continue
                if now - self.last_used(key) < self.grace_seconds:
                    continue
                with self.entry_lock(key, blocking=False) as free:
                    if not free:
                        continue
                    if can_evict is not None:
                        try:
                            if not can_evict(key):
                                continue
                        except Exception as e:
                            logger.warning(f"Not evicting {key[:12]}: {e}")
                            continue
                    self.remove_entry(key)
                total -= sizes[key]
                evicted.append(key)

            if evicted:
                logger.info(
                    f"Evicted {len(evicted)} local cache entries; "
                    f"{total / GB:.1f} GB of {self.max_bytes / GB:.1f} GB cap in use"
                )
            elif sizes and total + incoming_bytes > self.max_bytes:
                logger.warning(
                    f"Local cache over its cap ({total / GB:.1f} GB + "
                    f"{incoming_bytes / GB:.1f} GB incoming > {self.max_bytes / GB:.1f} GB) "
                    "but every entry is in use or recently used"
                )
        return evicted
