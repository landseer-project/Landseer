"""
MinIO object storage for Landseer artifacts.

Provides S3-compatible object storage for artifact caching and sharing
between workers.
"""

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, List, Optional, Union

from ..common import get_logger

logger = get_logger(__name__)

# Try to import minio, but don't fail if not installed
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception


@dataclass
class MinioConfig:
    """
    MinIO configuration.
    
    Can be configured via environment variables:
    - MINIO_ENDPOINT: MinIO server endpoint (default: localhost:9000)
    - MINIO_ACCESS_KEY: Access key (default: minioadmin)
    - MINIO_SECRET_KEY: Secret key (default: minioadmin)
    - MINIO_BUCKET: Default bucket name (default: landseer-artifacts)
    - MINIO_SECURE: Use HTTPS (default: false)
    """
    endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000")
    )
    access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    )
    secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin")
    )
    bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET", "landseer-artifacts")
    )
    secure: bool = field(
        default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    region: str = field(
        default_factory=lambda: os.getenv("MINIO_REGION", "us-east-1")
    )


class MinioStore:
    """
    MinIO object storage client for artifact management.
    
    Provides methods to:
    - Upload files and directories as artifacts
    - Download artifacts to local filesystem
    - List and query artifacts
    - Manage artifact metadata
    """
    
    def __init__(self, config: Optional[MinioConfig] = None):
        """
        Initialize MinIO store.
        
        Args:
            config: MinIO configuration
        """
        if not MINIO_AVAILABLE:
            logger.warning(
                "MinIO client not installed. Install with: pip install minio"
            )
            self._client = None
            self._available = False
            return
        
        self.config = config or MinioConfig()
        self._available = False
        
        try:
            self._client = Minio(
                self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region
            )
            self._ensure_bucket()
            self._available = True
            logger.info(f"MinIO store connected: {self.config.endpoint}")
        except Exception as e:
            logger.warning(f"Failed to connect to MinIO: {e}")
            self._client = None
    
    @property
    def is_available(self) -> bool:
        """Check if MinIO is available."""
        return self._available and self._client is not None
    
    def _ensure_bucket(self) -> None:
        """Ensure the artifact bucket exists."""
        if not self._client:
            return
        
        try:
            if not self._client.bucket_exists(self.config.bucket):
                self._client.make_bucket(self.config.bucket)
                logger.info(f"Created bucket: {self.config.bucket}")
        except S3Error as e:
            logger.error(f"Failed to create bucket: {e}")
            raise
    
    # =========================================================================
    # Upload Operations
    # =========================================================================
    
    def upload_file(
        self,
        local_path: Union[str, Path],
        object_key: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file to MinIO.
        
        Args:
            local_path: Path to local file
            object_key: Object key in bucket (e.g., "artifacts/task_1/model.pt")
            content_type: MIME type (auto-detected if not provided)
            metadata: Additional metadata
            
        Returns:
            True if upload successful
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping upload")
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return False
        
        try:
            self._client.fput_object(
                self.config.bucket,
                object_key,
                str(local_path),
                content_type=content_type or self._guess_content_type(local_path),
                metadata=metadata
            )
            logger.debug(f"Uploaded: {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload {object_key}: {e}")
            return False
    
    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload bytes data to MinIO.
        
        Args:
            data: Bytes to upload
            object_key: Object key in bucket
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            True if upload successful
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping upload")
            return False
        
        try:
            self._client.put_object(
                self.config.bucket,
                object_key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata=metadata
            )
            logger.debug(f"Uploaded bytes: {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Failed to upload bytes {object_key}: {e}")
            return False
    
    def upload_directory(
        self,
        local_dir: Union[str, Path],
        prefix: str,
        recursive: bool = True
    ) -> int:
        """
        Upload a directory to MinIO.
        
        Args:
            local_dir: Path to local directory
            prefix: Object key prefix (e.g., "artifacts/task_1/")
            recursive: Include subdirectories
            
        Returns:
            Number of files uploaded
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping upload")
            return 0
        
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            logger.error(f"Directory not found: {local_dir}")
            return 0
        
        uploaded = 0
        pattern = "**/*" if recursive else "*"
        
        for file_path in local_dir.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_dir)
                object_key = f"{prefix.rstrip('/')}/{rel_path.as_posix()}"
                
                if self.upload_file(file_path, object_key):
                    uploaded += 1
        
        logger.info(f"Uploaded {uploaded} files to {prefix}")
        return uploaded
    
    # =========================================================================
    # Download Operations
    # =========================================================================
    
    def download_file(
        self,
        object_key: str,
        local_path: Union[str, Path]
    ) -> bool:
        """
        Download a file from MinIO.
        
        Args:
            object_key: Object key in bucket
            local_path: Path to save file
            
        Returns:
            True if download successful
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping download")
            return False
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._client.fget_object(
                self.config.bucket,
                object_key,
                str(local_path)
            )
            logger.debug(f"Downloaded: {object_key} -> {local_path}")
            return True
        except S3Error as e:
            logger.error(f"Failed to download {object_key}: {e}")
            return False
    
    def download_bytes(self, object_key: str) -> Optional[bytes]:
        """
        Download an object as bytes.
        
        Args:
            object_key: Object key in bucket
            
        Returns:
            File contents as bytes, or None if failed
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping download")
            return None
        
        try:
            response = self._client.get_object(self.config.bucket, object_key)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"Failed to download bytes {object_key}: {e}")
            return None
    
    def download_directory(
        self,
        prefix: str,
        local_dir: Union[str, Path],
        recursive: bool = True
    ) -> int:
        """
        Download all objects with a prefix to a local directory.
        
        Args:
            prefix: Object key prefix
            local_dir: Local directory to save files
            recursive: Include nested prefixes
            
        Returns:
            Number of files downloaded
        """
        if not self.is_available:
            logger.warning("MinIO not available, skipping download")
            return 0
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        
        for obj in self.list_objects(prefix, recursive=recursive):
            if not obj.is_dir:
                # Get relative path from prefix
                rel_path = obj.object_name[len(prefix):].lstrip('/')
                local_path = local_dir / rel_path
                
                if self.download_file(obj.object_name, local_path):
                    downloaded += 1
        
        logger.info(f"Downloaded {downloaded} files from {prefix}")
        return downloaded
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def exists(self, object_key: str) -> bool:
        """
        Check if an object exists.
        
        Args:
            object_key: Object key in bucket
            
        Returns:
            True if object exists
        """
        if not self.is_available:
            return False
        
        try:
            self._client.stat_object(self.config.bucket, object_key)
            return True
        except S3Error:
            return False
    
    def get_metadata(self, object_key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata.
        
        Args:
            object_key: Object key in bucket
            
        Returns:
            Object metadata dict, or None if not found
        """
        if not self.is_available:
            return None
        
        try:
            stat = self._client.stat_object(self.config.bucket, object_key)
            return {
                "size": stat.size,
                "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "metadata": dict(stat.metadata) if stat.metadata else {}
            }
        except S3Error:
            return None
    
    def list_objects(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> Generator:
        """
        List objects with a prefix.
        
        Args:
            prefix: Object key prefix
            recursive: Include nested prefixes
            
        Yields:
            Object info objects
        """
        if not self.is_available:
            return
        
        try:
            objects = self._client.list_objects(
                self.config.bucket,
                prefix=prefix,
                recursive=recursive
            )
            yield from objects
        except S3Error as e:
            logger.error(f"Failed to list objects: {e}")
    
    def get_size(self, prefix: str = "") -> int:
        """
        Get total size of objects with a prefix.
        
        Args:
            prefix: Object key prefix
            
        Returns:
            Total size in bytes
        """
        total = 0
        for obj in self.list_objects(prefix):
            if not obj.is_dir:
                total += obj.size or 0
        return total
    
    # =========================================================================
    # Delete Operations
    # =========================================================================
    
    def delete(self, object_key: str) -> bool:
        """
        Delete an object.
        
        Args:
            object_key: Object key in bucket
            
        Returns:
            True if deletion successful
        """
        if not self.is_available:
            return False
        
        try:
            self._client.remove_object(self.config.bucket, object_key)
            logger.debug(f"Deleted: {object_key}")
            return True
        except S3Error as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            return False
    
    def delete_prefix(self, prefix: str) -> int:
        """
        Delete all objects with a prefix.
        
        Args:
            prefix: Object key prefix
            
        Returns:
            Number of objects deleted
        """
        if not self.is_available:
            return 0
        
        deleted = 0
        for obj in self.list_objects(prefix):
            if not obj.is_dir and self.delete(obj.object_name):
                deleted += 1
        
        logger.info(f"Deleted {deleted} objects with prefix: {prefix}")
        return deleted
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _guess_content_type(self, path: Path) -> str:
        """Guess content type based on file extension."""
        suffix = path.suffix.lower()
        types = {
            ".pt": "application/x-pytorch",
            ".pth": "application/x-pytorch",
            ".h5": "application/x-hdf5",
            ".hdf5": "application/x-hdf5",
            ".onnx": "application/x-onnx",
            ".npy": "application/x-numpy",
            ".npz": "application/x-numpy",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            ".txt": "text/plain",
            ".log": "text/plain",
            ".csv": "text/csv",
            ".pkl": "application/x-pickle",
            ".pickle": "application/x-pickle",
            ".tar": "application/x-tar",
            ".tar.gz": "application/gzip",
            ".tgz": "application/gzip",
            ".zip": "application/zip",
        }
        return types.get(suffix, "application/octet-stream")
    
    def get_artifact_key(self, cache_key: str, filename: str = "") -> str:
        """
        Generate object key for an artifact.
        
        Args:
            cache_key: Artifact cache key/hash
            filename: Optional filename within artifact
            
        Returns:
            Object key string
        """
        if filename:
            return f"artifacts/{cache_key}/{filename}"
        return f"artifacts/{cache_key}/"
    
    def artifact_exists(self, cache_key: str) -> bool:
        """
        Check if an artifact with the given cache key exists.
        
        Args:
            cache_key: Artifact cache key/hash
            
        Returns:
            True if artifact exists
        """
        prefix = self.get_artifact_key(cache_key)
        for _ in self.list_objects(prefix):
            return True
        return False


# Global store instance
_store: Optional[MinioStore] = None


def get_store() -> MinioStore:
    """
    Get the global MinIO store instance.
    
    Returns:
        MinioStore instance
        
    Raises:
        RuntimeError: If store not initialized
    """
    global _store
    if _store is None:
        raise RuntimeError("MinIO store not initialized. Call init_store() first.")
    return _store


def init_store(config: Optional[MinioConfig] = None) -> MinioStore:
    """
    Initialize the global MinIO store instance.
    
    Args:
        config: MinIO configuration
        
    Returns:
        MinioStore instance
    """
    global _store
    _store = MinioStore(config)
    return _store
