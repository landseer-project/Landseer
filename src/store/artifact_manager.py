"""
Artifact manager for coordinating artifact storage.

Provides high-level interface for:
- Uploading task artifacts to MinIO
- Downloading artifacts with local caching
- Tracking artifact metadata in database
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import get_logger
from .minio_store import MinioStore, MinioConfig

logger = get_logger(__name__)


@dataclass
class ArtifactInfo:
    """Information about an artifact."""
    cache_key: str
    task_id: str
    tool_name: str
    storage_type: str  # 'local', 'minio', 'both'
    local_path: Optional[Path] = None
    minio_key: Optional[str] = None
    size_bytes: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_hashes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cache_key": self.cache_key,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "storage_type": self.storage_type,
            "local_path": str(self.local_path) if self.local_path else None,
            "minio_key": self.minio_key,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "parent_hashes": self.parent_hashes,
            "metadata": self.metadata,
        }


class ArtifactManager:
    """
    High-level artifact management.
    
    Coordinates between local filesystem cache and MinIO storage
    to provide efficient artifact storage and retrieval.
    """
    
    def __init__(
        self,
        local_cache_dir: Path,
        minio_store: Optional[MinioStore] = None,
        use_minio: bool = True
    ):
        """
        Initialize artifact manager.
        
        Args:
            local_cache_dir: Directory for local artifact cache
            minio_store: MinIO store instance (creates new if not provided)
            use_minio: Whether to use MinIO for remote storage
        """
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_minio = use_minio
        if use_minio:
            self.minio = minio_store or MinioStore()
        else:
            self.minio = None
        
        # Track artifacts
        self._artifacts: Dict[str, ArtifactInfo] = {}
        
        logger.info(f"Artifact manager initialized. Local cache: {self.local_cache_dir}")
    
    # =========================================================================
    # Hash Computation
    # =========================================================================
    
    def compute_cache_key(
        self,
        task_id: str,
        tool_name: str,
        tool_image: str,
        config: Dict[str, Any],
        parent_hashes: List[str]
    ) -> str:
        """
        Compute cache key for a task based on its identity.
        
        Args:
            task_id: Task ID
            tool_name: Tool name
            tool_image: Container image
            config: Task configuration
            parent_hashes: Hashes of parent/dependency artifacts
            
        Returns:
            Cache key (hash string)
        """
        identity = {
            "tool_name": tool_name,
            "tool_image": tool_image,
            "config": config,
            "parents": sorted(parent_hashes)
        }
        
        json_str = json.dumps(identity, sort_keys=True, separators=(',', ':'))
        return hashlib.blake2s(json_str.encode()).hexdigest()
    
    # =========================================================================
    # Local Cache Operations
    # =========================================================================
    
    def get_local_path(self, cache_key: str) -> Path:
        """Get local cache path for a cache key."""
        return self.local_cache_dir / cache_key
    
    def local_exists(self, cache_key: str) -> bool:
        """Check if artifact exists in local cache."""
        path = self.get_local_path(cache_key)
        return path.exists() and (path / ".success").exists()
    
    def get_local_artifact(self, cache_key: str) -> Optional[Path]:
        """
        Get artifact from local cache if exists.
        
        Args:
            cache_key: Artifact cache key
            
        Returns:
            Path to local artifact directory, or None if not found
        """
        if self.local_exists(cache_key):
            return self.get_local_path(cache_key) / "output"
        return None
    
    def store_local(
        self,
        cache_key: str,
        source_dir: Path,
        task_id: str,
        tool_name: str,
        parent_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactInfo:
        """
        Store artifact in local cache.
        
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
        cache_dir = self.get_local_path(cache_key)
        output_dir = cache_dir / "output"
        
        # Create directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        if source_dir.exists():
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(source_dir, output_dir)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate size
        size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        
        # Write manifest
        manifest = {
            "cache_key": cache_key,
            "task_id": task_id,
            "tool_name": tool_name,
            "parent_hashes": parent_hashes or [],
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": size,
            "metadata": metadata or {}
        }
        (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        # Mark as complete
        (cache_dir / ".success").touch()
        
        # Create info
        info = ArtifactInfo(
            cache_key=cache_key,
            task_id=task_id,
            tool_name=tool_name,
            storage_type="local",
            local_path=output_dir,
            size_bytes=size,
            parent_hashes=parent_hashes or [],
            metadata=metadata or {}
        )
        
        self._artifacts[cache_key] = info
        logger.debug(f"Stored artifact locally: {cache_key[:12]}")
        return info
    
    # =========================================================================
    # MinIO Operations
    # =========================================================================
    
    def minio_exists(self, cache_key: str) -> bool:
        """Check if artifact exists in MinIO."""
        if not self.minio or not self.minio.is_available:
            return False
        return self.minio.artifact_exists(cache_key)
    
    def upload_to_minio(
        self,
        cache_key: str,
        source_dir: Optional[Path] = None
    ) -> bool:
        """
        Upload artifact to MinIO.
        
        Args:
            cache_key: Cache key for the artifact
            source_dir: Source directory (uses local cache if not provided)
            
        Returns:
            True if upload successful
        """
        if not self.minio or not self.minio.is_available:
            logger.debug("MinIO not available, skipping upload")
            return False
        
        # Use local cache if source not provided
        if source_dir is None:
            source_dir = self.get_local_path(cache_key) / "output"
        
        if not source_dir.exists():
            logger.error(f"Source directory not found: {source_dir}")
            return False
        
        # Upload directory
        prefix = self.minio.get_artifact_key(cache_key)
        count = self.minio.upload_directory(source_dir, prefix)
        
        # Upload manifest if exists
        manifest_path = source_dir.parent / "manifest.json"
        if manifest_path.exists():
            self.minio.upload_file(
                manifest_path,
                f"{prefix}manifest.json",
                content_type="application/json"
            )
        
        if count > 0:
            # Update artifact info
            if cache_key in self._artifacts:
                self._artifacts[cache_key].storage_type = "both"
                self._artifacts[cache_key].minio_key = prefix
            
            logger.info(f"Uploaded artifact to MinIO: {cache_key[:12]} ({count} files)")
            return True
        
        return False
    
    def download_from_minio(self, cache_key: str) -> Optional[Path]:
        """
        Download artifact from MinIO to local cache.
        
        Args:
            cache_key: Cache key for the artifact
            
        Returns:
            Path to local artifact directory, or None if not found
        """
        if not self.minio or not self.minio.is_available:
            return None
        
        if not self.minio_exists(cache_key):
            return None
        
        # Download to local cache
        cache_dir = self.get_local_path(cache_key)
        output_dir = cache_dir / "output"
        
        prefix = self.minio.get_artifact_key(cache_key)
        count = self.minio.download_directory(prefix, output_dir)
        
        if count > 0:
            # Mark as complete
            (cache_dir / ".success").touch()
            
            logger.info(f"Downloaded artifact from MinIO: {cache_key[:12]} ({count} files)")
            return output_dir
        
        return None
    
    # =========================================================================
    # High-Level Operations
    # =========================================================================
    
    def get_artifact(self, cache_key: str) -> Optional[Path]:
        """
        Get artifact, checking local cache first then MinIO.
        
        Args:
            cache_key: Cache key for the artifact
            
        Returns:
            Path to artifact output directory, or None if not found
        """
        # Check local cache first
        local_path = self.get_local_artifact(cache_key)
        if local_path:
            logger.debug(f"Cache hit (local): {cache_key[:12]}")
            return local_path
        
        # Try MinIO
        if self.use_minio:
            downloaded_path = self.download_from_minio(cache_key)
            if downloaded_path:
                logger.debug(f"Cache hit (MinIO): {cache_key[:12]}")
                return downloaded_path
        
        logger.debug(f"Cache miss: {cache_key[:12]}")
        return None
    
    def store_artifact(
        self,
        cache_key: str,
        source_dir: Path,
        task_id: str,
        tool_name: str,
        parent_hashes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        upload_to_remote: bool = True
    ) -> ArtifactInfo:
        """
        Store artifact in local cache and optionally MinIO.
        
        Args:
            cache_key: Cache key for the artifact
            source_dir: Directory containing output files
            task_id: Task ID
            tool_name: Tool name
            parent_hashes: Parent artifact hashes
            metadata: Additional metadata
            upload_to_remote: Whether to upload to MinIO
            
        Returns:
            ArtifactInfo object
        """
        # Store locally
        info = self.store_local(
            cache_key=cache_key,
            source_dir=source_dir,
            task_id=task_id,
            tool_name=tool_name,
            parent_hashes=parent_hashes,
            metadata=metadata
        )
        
        # Upload to MinIO if enabled
        if upload_to_remote and self.use_minio:
            if self.upload_to_minio(cache_key):
                info.storage_type = "both"
                info.minio_key = self.minio.get_artifact_key(cache_key)
        
        return info
    
    def artifact_exists(self, cache_key: str) -> bool:
        """
        Check if artifact exists anywhere.
        
        Args:
            cache_key: Cache key for the artifact
            
        Returns:
            True if artifact exists locally or in MinIO
        """
        return self.local_exists(cache_key) or self.minio_exists(cache_key)
    
    def get_parent_artifacts(
        self,
        parent_hashes: List[str]
    ) -> Dict[str, Optional[Path]]:
        """
        Get all parent artifacts.
        
        Args:
            parent_hashes: List of parent cache keys
            
        Returns:
            Dict mapping cache keys to paths (None if not found)
        """
        return {h: self.get_artifact(h) for h in parent_hashes}
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_local_stats(self) -> Dict[str, Any]:
        """Get local cache statistics."""
        total_size = 0
        artifact_count = 0
        
        for cache_dir in self.local_cache_dir.iterdir():
            if cache_dir.is_dir() and (cache_dir / ".success").exists():
                artifact_count += 1
                for f in cache_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
        
        return {
            "artifact_count": artifact_count,
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "cache_dir": str(self.local_cache_dir)
        }
    
    def get_minio_stats(self) -> Dict[str, Any]:
        """Get MinIO storage statistics."""
        if not self.minio or not self.minio.is_available:
            return {"available": False}
        
        total_size = self.minio.get_size("artifacts/")
        
        # Count artifacts
        artifact_count = 0
        seen_keys = set()
        for obj in self.minio.list_objects("artifacts/"):
            parts = obj.object_name.split("/")
            if len(parts) >= 2:
                cache_key = parts[1]
                if cache_key not in seen_keys:
                    seen_keys.add(cache_key)
                    artifact_count += 1
        
        return {
            "available": True,
            "artifact_count": artifact_count,
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "endpoint": self.minio.config.endpoint,
            "bucket": self.minio.config.bucket
        }
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
