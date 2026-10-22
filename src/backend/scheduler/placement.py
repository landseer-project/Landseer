"""
Tool-to-worker placement pins from ``configs/placement.yaml``.

A pinned tool's tasks go only to workers whose id starts with one of its
prefixes. The file is re-read when its mtime changes, so pins can be edited
while runs are in flight. ``LANDSEER_PLACEMENT_FILE`` overrides the path.
"""

import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from src.common import get_logger

logger = get_logger(__name__)

DEFAULT_PLACEMENT_FILE = "configs/placement.yaml"


def _token(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


class Placement:
    """Pins loaded from a YAML file, reloaded when the file changes."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.environ.get("LANDSEER_PLACEMENT_FILE", DEFAULT_PLACEMENT_FILE))
        self._mtime: Optional[float] = None
        self._pins: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def pins(self) -> Dict[str, List[str]]:
        """Current pins (tool token -> worker-id prefixes), reloading a changed file."""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        with self._lock:
            if mtime != self._mtime:
                self._mtime = mtime
                self._pins = self._load() if mtime is not None else {}
                logger.info(f"Placement pins from {self.path}: {self._pins or 'none'}")
            return self._pins

    def _load(self) -> Dict[str, List[str]]:
        try:
            data = yaml.safe_load(self.path.read_text()) or {}
        except Exception as e:
            # Keep scheduling with no pins rather than wedging every claim.
            logger.error(f"Ignoring unreadable placement file {self.path}: {e}")
            return {}
        pins = {}
        for tool, prefixes in (data.get("pins") or {}).items():
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            pins[_token(tool)] = [str(p) for p in prefixes or [] if str(p)]
        return pins

    def accepts(self, worker_id: Optional[str]) -> Callable[[object], bool]:
        """Predicate telling whether ``worker_id`` may take a task.

        An unknown worker (None) gets only unpinned tasks.
        """
        pins = self.pins()

        def accept(task) -> bool:
            prefixes = pins.get(_token(task.tool.name))
            if prefixes is None:
                return True
            return worker_id is not None and any(worker_id.startswith(p) for p in prefixes)

        return accept


_placement: Optional[Placement] = None


def get_placement() -> Placement:
    global _placement
    if _placement is None:
        _placement = Placement()
    return _placement
