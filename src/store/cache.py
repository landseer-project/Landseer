"""
Two-level caching for Landseer artifacts.

Implements the caching strategy described in OVERVIEWv1.md:
- Level 1: Local worker filesystem (fast, limited capacity)
- Level 2: AI Store / MinIO (slower, large capacity, shared across workers)

Cache operations:
1. Check local cache first
2. If miss, check MinIO
3. If found in MinIO, download to local cache
4. Store results in both levels
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import get_logger
from .minio_store import MinioStore, MinioConfig
from .artifact_manager import ArtifactManager, ArtifactInfo

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """
    Two-level cache configuration.
    
    Can be configured via environment variables:
    - LANDSEER_CACHE_DIR: Local cache directory
    - LANDSEER_CACHE_MAX_SIZE_GB: Maximum local cache size in GB
    - LANDSEER_USE_MINIO: Whether to use MinIO (default: true)
    """
    # Local cache settings
    local_cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "LANDSEER_CACHE_DIR",
            "/tmp/landseer_cache"
        ))
    )
    max_local_size_gb: float = field(
        default_factory=lambda: float(os.getenv(
            "LANDSEER_CACHE_MAX_SIZE_GB",
            "50"
        ))
    )
    
    # MinIO settings
    use_minio: bool = field(
        default_factory=lambda: os.getenv(
            "LANDSEER_USE_MINIO",
            "true"
        ).lower() == "true"
    )
    minio_config: Optional[MinioConfig] = None
    
    # Cache behavior
    auto_upload: bool = True  # Automatically upload to MinIO after storing
    eviction_threshold: float = 0.9  # Start eviction when cache reaches this % full


class TwoLevelCache:
    """
    Two-level caching implementation.
    
    Provides transparent caching across local filesystem and MinIO
    with automatic cache population and LRU eviction.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize two-level cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        # Initialize MinIO store
        minio_store = None
        if self.config.use_minio:
            minio_store = MinioStore(self.config.minio_config)
        
        # Initialize artifact manager
        self.manager = ArtifactManager(
            local_cache_dir=self.config.local_cache_dir,
            minio_store=minio_store,
            use_minio=self.config.use_minio
        )
        
        # Track access times for LRU eviction
        self._access_times: Dict[str, float] = {}
        
        logger.info(
            f"Two-level cache initialized. "
            f"Local: {self.config.local_cache_dir}, "
            f"MinIO: {'enabled' if self.config.use_minio else 'disabled'}"
        )
    
    # =========================================================================
    # Cache Operations
    # =========================================================================
    
    def get(self, cache_key: str) -> Optional[Path]:
        """
        Get artifact from cache.
        
        Checks local cache first, then MinIO. If found in MinIO,
        downloads to local cache for future access.
        
        Args:
            cache_key: Cache key for the artifact
            
        Returns:
            Path to artifact output directory, or None if not found
        """
        import time
        
        path = self.manager.get_artifact(cache_key)
        
        if path:
            # Update access time
            self._access_times[cache_key] = time.time()
        
        return path
    
    def put(
        self,
        cache_key: str,
        source_dir: Path,
        task_id: str,
        tool_name: str,
        parent_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactInfo:
        """
        Store artifact in cache.
        
        Stores locally and optionally uploads to MinIO based on config.
        Runs eviction if local cache is too full.
        
        Args:
            cache_key: Cache key for the artifact
            source_dir: Directory containing output files
            task_id: Task ID
            tool_name: Tool name
            parent_hashes: Parent artifact hashes
            metadata: Additional metadata
            
        Returns:
            ArtifactInfo object
        """
        import time
        
        # Check if eviction needed
        self._maybe_evict()
        
        # Store artifact
        info = self.manager.store_artifact(
            cache_key=cache_key,
            source_dir=source_dir,
            task_id=task_id,
            tool_name=tool_name,
            parent_hashes=parent_hashes,
            metadata=metadata,
            upload_to_remote=self.config.auto_upload
        )
        
        # Track access time
        self._access_times[cache_key] = time.time()
        
        return info
    
    def exists(self, cache_key: str) -> bool:
        """
        Check if artifact exists in cache.
        
        Args:
            cache_key: Cache key for the artifact
            
        Returns:
            True if artifact exists anywhere
        """
        return self.manager.artifact_exists(cache_key)
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Remove artifact from local cache.
        
        Note: Does not remove from MinIO to allow other workers to use it.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if removed
        """
        local_path = self.manager.get_local_path(cache_key)
        
        if local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)
            self._access_times.pop(cache_key, None)
            logger.debug(f"Invalidated local cache: {cache_key[:12]}")
            return True
        
        return False
    
    # =========================================================================
    # Eviction
    # =========================================================================
    
    def _get_local_size(self) -> int:
        """Get current local cache size in bytes."""
        total = 0
        for cache_dir in self.config.local_cache_dir.iterdir():
            if cache_dir.is_dir():
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return total
    
    def _maybe_evict(self) -> int:
        """
        Evict old entries if cache is too full.
        
        Uses LRU eviction to remove least recently used entries.
        
        Returns:
            Number of entries evicted
        """
        max_size = int(self.config.max_local_size_gb * 1024 * 1024 * 1024)
        threshold = int(max_size * self.config.eviction_threshold)
        
        current_size = self._get_local_size()
        
        if current_size < threshold:
            return 0
        
        logger.info(
            f"Cache eviction triggered. "
            f"Size: {self._format_size(current_size)} / {self._format_size(max_size)}"
        )
        
        evicted = 0
        target_size = int(max_size * 0.7)  # Evict down to 70%
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self._access_times.keys(),
            key=lambda k: self._access_times.get(k, 0)
        )
        
        for cache_key in sorted_keys:
            if current_size <= target_size:
                break
            
            local_path = self.manager.get_local_path(cache_key)
            if not local_path.exists():
                continue
            
            # Calculate size before removal
            entry_size = sum(
                f.stat().st_size
                for f in local_path.rglob("*")
                if f.is_file()
            )
            
            # Ensure it's uploaded to MinIO before evicting
            if self.config.use_minio and not self.manager.minio_exists(cache_key):
                self.manager.upload_to_minio(cache_key)
            
            # Remove local copy
            shutil.rmtree(local_path, ignore_errors=True)
            self._access_times.pop(cache_key, None)
            
            current_size -= entry_size
            evicted += 1
            logger.debug(f"Evicted: {cache_key[:12]}")
        
        logger.info(
            f"Evicted {evicted} entries. "
            f"New size: {self._format_size(current_size)}"
        )
        
        return evicted
    
    def clear_local(self) -> int:
        """
        Clear all local cache entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        
        for cache_dir in self.config.local_cache_dir.iterdir():
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed += 1
        
        self._access_times.clear()
        logger.info(f"Cleared local cache: {removed} entries")
        return removed
    
    # =========================================================================
    # Pre-fetching
    # =========================================================================
    
    def prefetch(self, cache_keys: List[str]) -> int:
        """
        Pre-fetch artifacts from MinIO to local cache.
        
        Useful for warming the cache before task execution.
        
        Args:
            cache_keys: List of cache keys to prefetch
            
        Returns:
            Number of artifacts fetched
        """
        fetched = 0
        
        for cache_key in cache_keys:
            if self.manager.local_exists(cache_key):
                continue
            
            if self.manager.download_from_minio(cache_key):
                fetched += 1
        
        if fetched > 0:
            logger.info(f"Prefetched {fetched} artifacts from MinIO")
        
        return fetched
    
    def prefetch_parents(self, parent_hashes: List[str]) -> Dict[str, Optional[Path]]:
        """
        Pre-fetch all parent artifacts.
        
        Args:
            parent_hashes: List of parent cache keys
            
        Returns:
            Dict mapping cache keys to paths
        """
        self.prefetch(parent_hashes)
        return self.manager.get_parent_artifacts(parent_hashes)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "local": self.manager.get_local_stats(),
            "minio": self.manager.get_minio_stats(),
            "config": {
                "max_local_size_gb": self.config.max_local_size_gb,
                "eviction_threshold": self.config.eviction_threshold,
                "auto_upload": self.config.auto_upload
            }
        }
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
