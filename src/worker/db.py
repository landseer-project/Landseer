"""
Artifact cache database for Landseer worker.

This module provides content-addressable caching for task artifacts:
- Compute stable hashes for tasks based on tool identity and dependencies
- Store and retrieve cached artifacts
- Track artifact metadata and provenance

The cache is designed to be shared across workers and runs to maximize reuse.
"""

import hashlib
import json
import os
import shutil
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import get_logger
from .client import TaskInfo

logger = get_logger(__name__)


def _stable_json_hash(obj: Any) -> str:
    """
    Compute a stable hash from a JSON-serializable object.
    
    Uses deterministic serialization (sorted keys, no extra spaces)
    and BLAKE2s for fast hashing.
    """
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2s(data).hexdigest()


def _hash_file(file_path: str) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex-encoded hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


@dataclass
class ArtifactMetadata:
    """Metadata about a cached artifact."""
    node_hash: str
    task_id: str
    tool_name: str
    tool_image: str
    parent_hashes: List[str]
    created_at: str
    execution_time_ms: int
    files: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_hash": self.node_hash,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "tool_image": self.tool_image,
            "parent_hashes": self.parent_hashes,
            "created_at": self.created_at,
            "execution_time_ms": self.execution_time_ms,
            "files": self.files
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        """Create from dictionary."""
        return cls(
            node_hash=data["node_hash"],
            task_id=data["task_id"],
            tool_name=data["tool_name"],
            tool_image=data["tool_image"],
            parent_hashes=data.get("parent_hashes", []),
            created_at=data["created_at"],
            execution_time_ms=data.get("execution_time_ms", 0),
            files=data.get("files", [])
        )


class ArtifactCacheDB:
    """
    Content-addressable artifact cache.
    
    Directory layout:
        <root>/<node_hash>/
            output/       # Tool-produced files
            manifest.json # Artifact metadata
            .success      # Marker indicating successful completion
    
    Properties:
        - Append-only: Cache hits never modify existing files
        - Hash stability: Depends on parent hashes + tool identity
        - Thread-safe: Per-hash locks prevent concurrent writes
    """
    
    def __init__(self, root: Path):
        """
        Initialize the artifact cache.
        
        Args:
            root: Root directory for the cache
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Per-hash locks for thread safety
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        
        logger.info(f"Artifact cache initialized at: {self.root}")
    
    # =========================================================================
    # Hash Computation
    # =========================================================================
    
    def compute_tool_hash(self, task: TaskInfo) -> str:
        """
        Compute a hash that identifies the tool configuration.
        
        This includes:
        - Tool name
        - Container image
        - Command
        - Configuration parameters
        
        Args:
            task: Task information
            
        Returns:
            Tool identity hash
        """
        identity = {
            "name": task.tool_name,
            "image": task.tool_image,
            "command": task.tool_command,
            "config": task.config,
            "task_type": task.task_type
        }
        return _stable_json_hash(identity)
    
    def compute_task_hash(
        self,
        task: TaskInfo,
        parent_hashes: List[str]
    ) -> str:
        """
        Compute a hash for a task based on its tool and dependencies.
        
        The hash is stable across runs if:
        - Same tool configuration
        - Same parent artifacts
        
        Args:
            task: Task information
            parent_hashes: Hashes of parent/dependency artifacts
            
        Returns:
            Node hash for cache lookup
        """
        tool_hash = self.compute_tool_hash(task)
        node_data = {
            "parents": sorted(parent_hashes),  # Sort for determinism - Check what's this we need to keep order of the parents intact - eg. ABC and ACB should be different
            "tool": tool_hash
        }
        return _stable_json_hash(node_data)
    
    def compute_data_hash(self, data_path: Path, variant: str = "clean") -> str:
        """
        Compute a hash for input data (dataset/model).
        
        Args:
            data_path: Path to data directory or file
            variant: Data variant (e.g., "clean", "poisoned")
            
        Returns:
            Data hash
        """
        data_info = {"variant": variant, "path": str(data_path)}
        
        # If it's a file, include its hash
        if data_path.is_file():
            try:
                data_info["file_hash"] = _hash_file(str(data_path))
            except Exception:
                pass
        # If it's a directory, include a summary of its contents
        elif data_path.is_dir():
            try:
                # Hash based on file names and sizes (fast approximation)
                files = []
                for p in sorted(data_path.rglob("*")):
                    if p.is_file():
                        files.append({
                            "path": p.relative_to(data_path).as_posix(),
                            "size": p.stat().st_size
                        })
                data_info["files_summary"] = files[:100]  # Limit for performance
            except Exception:
                pass
        
        return _stable_json_hash(data_info)
    
    # =========================================================================
    # Cache Operations
    # =========================================================================
    
    def path_for(self, node_hash: str) -> Path:
        """Get the directory path for a node hash."""
        return self.root / node_hash
    
    def output_path_for(self, node_hash: str) -> Path:
        """Get the output directory path for a node hash."""
        return self.path_for(node_hash) / "output"
    
    def exists(self, node_hash: str) -> bool:
        """
        Check if an artifact exists in the cache.
        
        An artifact exists if its success marker is present.
        
        Args:
            node_hash: Artifact hash
            
        Returns:
            True if artifact exists and is complete
        """
        return (self.path_for(node_hash) / ".success").exists()
    
    def get_cached_artifact(self, node_hash: str) -> Optional[Path]:
        """
        Get the path to a cached artifact if it exists.
        
        Args:
            node_hash: Artifact hash
            
        Returns:
            Path to output directory if cached, None otherwise
        """
        if self.exists(node_hash):
            return self.output_path_for(node_hash)
        return None
    
    def get_metadata(self, node_hash: str) -> Optional[ArtifactMetadata]:
        """
        Get metadata for a cached artifact.
        
        Args:
            node_hash: Artifact hash
            
        Returns:
            ArtifactMetadata if cached, None otherwise
        """
        manifest_path = self.path_for(node_hash) / "manifest.json"
        if not manifest_path.exists():
            return None
        
        try:
            data = json.loads(manifest_path.read_text())
            return ArtifactMetadata.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to read manifest for {node_hash}: {e}")
            return None
    
    # =========================================================================
    # Locking
    # =========================================================================
    
    def _get_lock(self, node_hash: str) -> threading.Lock:
        """Get or create a lock for a node hash."""
        with self._global_lock:
            if node_hash not in self._locks:
                self._locks[node_hash] = threading.Lock()
            return self._locks[node_hash]
    
    def lock(self, node_hash: str) -> threading.Lock:
        """
        Acquire a lock for a node hash.
        
        The lock is acquired before returning. Caller must release it.
        
        Args:
            node_hash: Artifact hash
            
        Returns:
            Acquired lock
        """
        lock = self._get_lock(node_hash)
        lock.acquire()
        return lock
    
    # =========================================================================
    # Storage
    # =========================================================================
    
    def store_artifact(
        self,
        node_hash: str,
        output_path: Path,
        task: TaskInfo,
        execution_time_ms: int = 0,
        parent_hashes: Optional[List[str]] = None
    ) -> bool:
        """
        Store an artifact in the cache.
        
        Args:
            node_hash: Artifact hash
            output_path: Path to output directory to cache
            task: Task information
            execution_time_ms: Execution time in milliseconds
            parent_hashes: Parent artifact hashes
            
        Returns:
            True if stored successfully
        """
        lock = self.lock(node_hash)
        try:
            # Check if already exists (race condition protection)
            if self.exists(node_hash):
                logger.debug(f"Artifact {node_hash[:12]} already exists, skipping store")
                return True
            
            node_dir = self.path_for(node_hash)
            cache_output = node_dir / "output"
            
            # Create node directory
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy output files
            if output_path.exists():
                if output_path.is_dir():
                    shutil.copytree(output_path, cache_output, dirs_exist_ok=True)
                else:
                    cache_output.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, cache_output / output_path.name)
            else:
                cache_output.mkdir(parents=True, exist_ok=True)
            
            # Build file manifest
            files = []
            for f in cache_output.rglob("*"):
                if f.is_file():
                    try:
                        rel_path = f.relative_to(node_dir).as_posix()
                        files.append({
                            "rel": rel_path,
                            "size": f.stat().st_size
                        })
                    except Exception:
                        pass
            
            # Create metadata
            metadata = ArtifactMetadata(
                node_hash=node_hash,
                task_id=task.id,
                tool_name=task.tool_name,
                tool_image=task.tool_image,
                parent_hashes=parent_hashes or [],
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                execution_time_ms=execution_time_ms,
                files=files
            )
            
            # Write manifest
            manifest_path = node_dir / "manifest.json"
            manifest_path.write_text(json.dumps(metadata.to_dict(), indent=2))
            
            # Mark as complete
            success_marker = node_dir / ".success"
            success_marker.touch()
            
            # Make files read-only to prevent accidental modification
            self._make_readonly(node_dir)
            
            logger.info(f"Stored artifact {node_hash[:12]} ({task.tool_name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store artifact {node_hash[:12]}: {e}")
            return False
        finally:
            lock.release()
    
    def _make_readonly(self, directory: Path) -> None:
        """Make all files in a directory read-only."""
        try:
            for p in directory.rglob("*"):
                if p.is_file():
                    p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except Exception as e:
            logger.debug(f"Failed to make files read-only: {e}")
    
    def mark_failed(
        self,
        node_hash: str,
        error_message: str,
        task: TaskInfo
    ) -> None:
        """
        Mark an artifact as failed.
        
        This creates a failure marker with the error message,
        which can be used for debugging and to avoid retrying
        known-bad configurations.
        
        Args:
            node_hash: Artifact hash
            error_message: Error message describing the failure
            task: Task information
        """
        lock = self.lock(node_hash)
        try:
            node_dir = self.path_for(node_hash)
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Write failure info
            failure_path = node_dir / ".failed"
            failure_path.touch()
            
            reason_path = node_dir / "failure_reason.txt"
            reason_path.write_text(f"Task: {task.id}\nTool: {task.tool_name}\n\nError:\n{error_message}")
            
        except Exception as e:
            logger.warning(f"Failed to mark failure for {node_hash[:12]}: {e}")
        finally:
            lock.release()
    
    def is_failed(self, node_hash: str) -> bool:
        """Check if an artifact is marked as failed."""
        return (self.path_for(node_hash) / ".failed").exists()
    
    def get_failure_reason(self, node_hash: str) -> Optional[str]:
        """Get the failure reason for a failed artifact."""
        reason_path = self.path_for(node_hash) / "failure_reason.txt"
        if reason_path.exists():
            return reason_path.read_text()
        return None
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        artifact_count = 0
        failed_count = 0
        
        try:
            for node_dir in self.root.iterdir():
                if node_dir.is_dir():
                    if (node_dir / ".success").exists():
                        artifact_count += 1
                        # Calculate size
                        for f in node_dir.rglob("*"):
                            if f.is_file():
                                total_size += f.stat().st_size
                    elif (node_dir / ".failed").exists():
                        failed_count += 1
        except Exception as e:
            logger.warning(f"Error calculating cache stats: {e}")
        
        return {
            "artifact_count": artifact_count,
            "failed_count": failed_count,
            "total_size_bytes": total_size,
            "total_size_human": self._format_size(total_size),
            "cache_root": str(self.root)
        }
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} PB"
    
    def cleanup_failed(self) -> int:
        """
        Remove all failed artifact entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        try:
            for node_dir in self.root.iterdir():
                if node_dir.is_dir() and (node_dir / ".failed").exists():
                    shutil.rmtree(node_dir, ignore_errors=True)
                    removed += 1
        except Exception as e:
            logger.warning(f"Error cleaning up failed artifacts: {e}")
        
        logger.info(f"Removed {removed} failed artifact entries")
        return removed
    
    def verify_artifact(self, node_hash: str) -> bool:
        """
        Verify that a cached artifact is complete and valid.
        
        Args:
            node_hash: Artifact hash
            
        Returns:
            True if artifact is valid
        """
        node_dir = self.path_for(node_hash)
        
        # Check basic structure
        if not node_dir.exists():
            return False
        if not (node_dir / ".success").exists():
            return False
        if not (node_dir / "manifest.json").exists():
            return False
        if not (node_dir / "output").exists():
            return False
        
        # Verify manifest can be read
        try:
            metadata = self.get_metadata(node_hash)
            if metadata is None:
                return False
        except Exception:
            return False
        
        return True


class CacheManager:
    """
    High-level cache management for workers.
    
    Provides a simplified interface for cache operations
    during task execution.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for the artifact cache
        """
        self.db = ArtifactCacheDB(cache_dir)
        self._current_pipeline_hashes: Dict[str, List[str]] = {}
    
    def check_cache(
        self,
        task: TaskInfo,
        parent_hashes: Optional[List[str]] = None
    ) -> Optional[Path]:
        """
        Check if a task's output is cached.
        
        Args:
            task: Task to check
            parent_hashes: Parent artifact hashes
            
        Returns:
            Path to cached output if available, None otherwise
        """
        node_hash = self.db.compute_task_hash(task, parent_hashes or [])
        return self.db.get_cached_artifact(node_hash)
    
    def store_result(
        self,
        task: TaskInfo,
        output_path: Path,
        execution_time_ms: int,
        parent_hashes: Optional[List[str]] = None
    ) -> str:
        """
        Store task result in cache.
        
        Args:
            task: Completed task
            output_path: Path to output directory
            execution_time_ms: Execution time in milliseconds
            parent_hashes: Parent artifact hashes
            
        Returns:
            Node hash of stored artifact
        """
        node_hash = self.db.compute_task_hash(task, parent_hashes or [])
        self.db.store_artifact(
            node_hash=node_hash,
            output_path=output_path,
            task=task,
            execution_time_ms=execution_time_ms,
            parent_hashes=parent_hashes
        )
        return node_hash
    
    def record_failure(
        self,
        task: TaskInfo,
        error_message: str,
        parent_hashes: Optional[List[str]] = None
    ) -> str:
        """
        Record a task failure in cache.
        
        Args:
            task: Failed task
            error_message: Error message
            parent_hashes: Parent artifact hashes
            
        Returns:
            Node hash of failed artifact
        """
        node_hash = self.db.compute_task_hash(task, parent_hashes or [])
        self.db.mark_failed(node_hash, error_message, task)
        return node_hash
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.db.get_cache_stats()
