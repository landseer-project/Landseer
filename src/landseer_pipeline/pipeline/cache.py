import hashlib
import json
from pathlib import Path
from typing import List

from filelock import FileLock
from landseer_pipeline.utils.docker import get_image_digest

class CacheManager:
    def __init__(self, settings):
        self.settings = settings

    def safe_cache_path(self, cache_key):
        cache_path = self.get_cache_path(cache_key)
        lock_path = cache_path / ".lock"
        lock = FileLock(str(lock_path))
        lock.acquire()
        return cache_path, lock

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the path to the cache directory for a given cache key."""
        return self.settings.output_dir / cache_key

    def is_cached(self, cache_key: str) -> bool:
        """Check if the output for the given cache key exists."""
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        success_file = cache_path / ".success"
        if success_file.exists():
            return True

    def compute_cache_key(
        self, prev_cache_key: str, current_tool, stage: str, input_path: str, dataset
    ) -> str:
        data = {
            "tool_sequence": prev_cache_key,  # order matters
            "current_tool": str(current_tool),
            "stage": stage,
            "input_path": str(input_path),
            "dataset": str(dataset),
        }
        json_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.sha256(json_str.encode()).hexdigest()
        cache_dir = self.settings.output_dir / hash_val

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(data, f, indent=4)
        return hash_val

    def get_cached_output(self, cache_key: str) -> str:
        cache_path = self.settings.output_dir / cache_key / "output"
        return cache_path if cache_path.exists() else None

    def mark_as_cached(self, cache_key: str, output_path: str):
        """Mark the output as cached by creating a .success file."""
        cache_path = self.get_cache_path(cache_key)
        success_file = cache_path / ".success"
        success_file.touch()

    def mark_as_failed(self, cache_key: str):
        """Mark the output as failed by creating a .failed file."""
        try:
            cache_path = self.get_cache_path(cache_key)
            cache_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            failed_file = cache_path / ".failed"
            failed_file.touch()
        except Exception as e:
            # Log the error but don't fail the whole process
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to mark cache as failed for {cache_key}: {e}")
