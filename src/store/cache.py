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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import get_logger
from .minio_store import MinioStore, MinioConfig
from .artifact_manager import ArtifactManager, ArtifactInfo
from .local_lru import GB

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    """
    Two-level cache configuration.
    
    Env:
    - LANDSEER_CACHE_DIR: Local cache directory
    - LANDSEER_CACHE_MAX_SIZE_GB: Local cap; least recently used entries are
      evicted past it (default: min(500 GB, 40% of the cache filesystem))
    - LANDSEER_CACHE_KEEP_LOCAL: Keep entries locally after upload (default 1)
    """
    local_cache_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "LANDSEER_CACHE_DIR",
            "/data/landseer/cache"
        ))
    )
    # None -> local_lru.default_max_bytes (reads LANDSEER_CACHE_MAX_SIZE_GB)
    max_local_size_gb: Optional[float] = None
    keep_local: bool = field(
        default_factory=lambda: os.getenv("LANDSEER_CACHE_KEEP_LOCAL", "1").lower()
        not in ("0", "false", "no")
    )
    minio_config: Optional[MinioConfig] = None


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
        
        minio_store = MinioStore(self.config.minio_config)
        max_bytes = (
            int(self.config.max_local_size_gb * GB)
            if self.config.max_local_size_gb is not None
            else None
        )
        self.manager = ArtifactManager(
            local_cache_dir=self.config.local_cache_dir,
            minio_store=minio_store,
            use_minio=True,
            max_local_bytes=max_bytes,
            keep_local=self.config.keep_local,
        )

        logger.info(
            f"Two-level cache initialized. "
            f"Local: {self.config.local_cache_dir} "
            f"(cap {self.manager.lru.max_bytes / GB:.0f} GB, "
            f"keep local: {self.config.keep_local}), MinIO: enabled"
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
        return self.manager.get_artifact(cache_key)
    
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
        # Eviction (if needed) happens inside the manager before the write.
        return self.manager.store_artifact(
            cache_key=cache_key,
            source_dir=source_dir,
            task_id=task_id,
            tool_name=tool_name,
            parent_hashes=parent_hashes,
            metadata=metadata,
        )
    
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
        return self.manager.delete_local(cache_key)

    # =========================================================================
    # Eviction
    # =========================================================================

    def _get_local_size(self) -> int:
        """Bytes held by complete local entries (datasets/ and staging dirs excluded)."""
        return self.manager.lru.total_size()

    def _maybe_evict(self) -> int:
        """Evict least-recently-used entries if the local cache is over its cap."""
        return len(self.manager.lru.ensure_space(can_evict=self.manager._durable))

    def clear_local(self) -> int:
        """
        Clear all local cache entries (datasets/ is kept).

        Returns:
            Number of entries removed
        """
        removed = 0
        for cache_key in self.manager.lru.entries():
            if self.manager.delete_local(cache_key):
                removed += 1
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
                "max_local_size_gb": self.manager.lru.max_bytes / GB,
                "keep_local": self.config.keep_local,
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
