import hashlib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
#TODO: Make a class for cahcing to get output path from settings
class CacheManager:
    def __init__(self, settings):
        self.settings = settings

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the path to the cache directory for a given cache key."""
        return self.settings.data_dir / cache_key
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if the output for the given cache key exists."""
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        # Check if .success file exists or if the directory is not empty
        success_file = cache_path / ".success"
        if success_file.exists():
            return True
        
    def is_clean_cached(self, cache_key: str) -> bool:
        """Check if the clean output for the given cache key exists."""
        cache_path = self.get_cache_path(cache_key)
        clean_output_path = cache_path / "clean"
        return clean_output_path.exists() and clean_output_path.is_dir() and any(clean_output_path.iterdir())
    
    def is_poisoned_cached(self, cache_key: str) -> bool:
        """Check if the poisoned output for the given cache key exists."""
        cache_path = self.get_cache_path(cache_key)
        poisoned_output_path = cache_path / "poisoned"
        return poisoned_output_path.exists() and poisoned_output_path.is_dir() and any(poisoned_output_path.iterdir())
    
    def mark_as_success(self, cache_key: str):
        """Mark the cache as successfully created by creating a .success file."""
        cache_path = self.get_cache_path(cache_key)
        cache_path.mkdir(parents=True, exist_ok=True)
        success_file = cache_path / ".success"
        success_file.touch()
    
    