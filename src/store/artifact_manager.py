"""
Artifact manager for coordinating artifact storage.

Provides high-level interface for:
- Uploading task artifacts to MinIO
- Downloading artifacts with local caching
- Tracking artifact metadata in database

Dataset de-duplication: many tools copy their whole input dataset into their
output (CelebA: ~2.6 GB, 200k files per task). A file whose relative path,
size and sha256 equal the same entry in the run's dataset manifest is not
uploaded; the completion marker still lists it, with ``source: dataset`` and
the dataset's ``dataset_content_id``. A download rebuilds such files from this
worker's verified local copy of that dataset (hard link, else copy). Readers
that predate this see them as missing and treat the artifact as a miss.
Set LANDSEER_ARTIFACT_DATASET_DEDUP=0 to upload everything.
"""

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from ..common import dataset_manifest as dsm
from ..common import get_logger
from ..common.fsutil import clone_tree, hardlink_or_copy, make_tree_read_only
from .local_lru import LocalLRU
from .minio_store import (
    ARTIFACT_COMPLETE_MARKER,
    ARTIFACT_MANIFEST,
    MinioStore,
    MinioConfig,
)

logger = get_logger(__name__)

# Marker ``source`` of a file left out of the upload because the dataset holds it.
DATASET_SOURCE = "dataset"


def dataset_dedup_enabled() -> bool:
    return os.environ.get("LANDSEER_ARTIFACT_DATASET_DEDUP", "1").lower() not in ("0", "false", "no")


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
        use_minio: bool = True,
        max_local_bytes: Optional[int] = None,
        keep_local: bool = True,
    ):
        """
        Initialize artifact manager.

        Args:
            local_cache_dir: Directory for local artifact cache
            minio_store: MinIO store instance (creates new if not provided)
            use_minio: Whether to use MinIO for remote storage
            max_local_bytes: Local cache cap (default: see local_lru.default_max_bytes)
            keep_local: Keep entries locally after upload, evicting LRU past the cap
        """
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.lru = LocalLRU(self.local_cache_dir, max_bytes=max_local_bytes)
        self.keep_local = keep_local
        
        self.use_minio = use_minio
        if use_minio:
            self.minio = minio_store or MinioStore()
        else:
            self.minio = None
        
        # Track artifacts
        self._artifacts: Dict[str, ArtifactInfo] = {}

        # content_id -> (verified local dataset dir, its manifest, where it is
        # published: name/variant/minio_key). See register_dataset.
        self._datasets: Dict[str, Tuple[Path, Dict[str, Any], Dict[str, Any]]] = {}
        # Dataset whose files uploads leave out (the active run's).
        self._active_dataset: Optional[str] = None
        # (content_id, published location from the marker) -> local verified
        # dir, for datasets not registered here (the worker syncs from MinIO).
        self.dataset_resolver: Optional[Callable[[str, Dict[str, Any]], Optional[Path]]] = None
        
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
    # Dataset De-duplication
    # =========================================================================

    def register_dataset(
        self,
        path: Path,
        manifest: Dict[str, Any],
        source: Optional[Dict[str, Any]] = None,
        active: bool = True,
    ) -> None:
        """
        Record a local dataset dir already verified against ``manifest``.

        Downloads rebuild left-out files from it; with ``active`` uploads also
        leave out files identical to its manifest entries.
        """
        content_id = manifest.get("content_id")
        if not content_id or not isinstance(manifest.get("files"), dict):
            return
        self._datasets[content_id] = (Path(path), manifest, dict(source or {}))
        if active:
            self._active_dataset = content_id

    def set_active_dataset(self, content_id: Optional[str]) -> None:
        """Dataset uploads leave files out for (None: upload everything)."""
        self._active_dataset = content_id if content_id in self._datasets else None

    def _dataset_copies(self, files: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """rel -> dataset content_id for files identical (path, size, sha256) to the active dataset's."""
        if not dataset_dedup_enabled() or self._active_dataset not in self._datasets:
            return {}
        content_id = self._active_dataset
        dataset_files = self._datasets[content_id][1]["files"]
        copies = {}
        for rel, entry in files.items():
            # manifest.json shares its object key with the cache manifest.
            if rel == ARTIFACT_MANIFEST:
                continue
            ds = dataset_files.get(rel)
            if (
                isinstance(ds, dict)
                and int(ds.get("size", -1)) == entry["size"]
                and ds.get("sha256") == entry["sha256"]
            ):
                copies[rel] = content_id
        return copies

    def _dataset_dir(self, content_id: str, location: Dict[str, Any]) -> Optional[Path]:
        """Local verified copy of dataset content_id, or None."""
        known = self._datasets.get(content_id)
        if known and known[0].is_dir():
            return known[0]
        if self.dataset_resolver is not None:
            try:
                return self.dataset_resolver(content_id, location)
            except Exception as e:
                logger.warning(f"Could not get dataset {content_id[:12]} locally: {e}")
        return None

    def _restore_dataset_files(
        self,
        output: Path,
        expected: Dict[str, Dict[str, Any]],
        datasets: Dict[str, Dict[str, Any]],
    ) -> Set[str]:
        """
        Rebuild the files the marker says were left out, from local dataset copies.

        Hard-linked (copied across filesystems). Each source must match the
        marker's size and sha256; sha256 comes from the dataset dir's digest
        sidecar (keyed by size and mtime, as for dataset verification), so a
        stale sidecar is the only way a changed file could pass. Raises when a
        dataset or file is unavailable, which makes the download a miss.
        """
        by_dataset: Dict[str, List[str]] = {}
        for rel, entry in expected.items():
            if entry.get("source") == DATASET_SOURCE:
                parts = Path(rel).parts
                if not parts or Path(rel).is_absolute() or ".." in parts:
                    raise RuntimeError(f"unsafe dataset file path in marker: {rel!r}")
                by_dataset.setdefault(str(entry.get("dataset_content_id")), []).append(rel)
        methods: Dict[str, int] = {}
        for content_id, rels in by_dataset.items():
            root = self._dataset_dir(content_id, datasets.get(content_id) or {})
            if root is None:
                raise RuntimeError(f"dataset {content_id[:12]} is not available on this worker")
            actual = dsm.hash_files(root, rels)
            bad = [
                rel for rel in rels
                if actual.get(rel) != {"size": expected[rel]["size"], "sha256": expected[rel].get("sha256")}
            ]
            if bad:
                raise RuntimeError(
                    f"dataset {content_id[:12]} at {root} does not hold {len(bad)} files (e.g. {bad[:3]})"
                )
            for rel in rels:
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                method = hardlink_or_copy(root / rel, dest)
                methods[method] = methods.get(method, 0) + 1
        if methods:
            logger.info(f"Rebuilt dataset files from local copies: {methods}")
        return {rel for rels in by_dataset.values() for rel in rels}

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
            self.lru.touch(cache_key)
            return self.get_local_path(cache_key) / "output"
        return None

    @contextmanager
    def _entry_lock(self, cache_key: str) -> Iterator[None]:
        """Serialize writers of one cache entry on this host (flock).

        Eviction takes the same lock, so an entry being written or read here
        is never removed underneath us.
        """
        with self.lru.entry_lock(cache_key):
            yield

    def _durable(self, cache_key: str) -> bool:
        """Safe to drop locally: a complete copy is in MinIO (uploading it if needed)."""
        if not self.use_minio:
            return True
        return self.minio_exists(cache_key) or self.upload_to_minio(cache_key)

    def _make_room(self, incoming_bytes: int, protect: Optional[str] = None) -> None:
        """Evict least-recently-used local entries if this write would pass the cap."""
        try:
            self.lru.ensure_space(incoming_bytes, can_evict=self._durable, protect=protect)
        except Exception as e:
            logger.warning(f"Local cache eviction failed: {e}")

    def should_keep_local(self, cache_key: str) -> bool:
        """Keep this entry on local disk after use (False if keep-local is off or it exceeds the cap)."""
        return self.keep_local and self.lru.fits(self.lru.entry_size(cache_key))

    def _staging_dir(self, cache_key: str) -> Path:
        """Private temp dir next to the final entry (same filesystem, so rename is atomic)."""
        return self.local_cache_dir / f".tmp-{cache_key}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    def _publish_local(self, staging: Path, cache_key: str) -> Path:
        """Move a fully written staging dir into place as the final entry."""
        final = self.get_local_path(cache_key)
        if final.exists() and (final / ".success").exists():
            # A complete copy is already published; keep it.
            shutil.rmtree(staging, ignore_errors=True)
            return final
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        return final

    @staticmethod
    def _list_files(root: Path) -> Dict[str, int]:
        """Map of posix relative path -> size for every regular file under root."""
        if not root.is_dir():
            return {}
        return {
            p.relative_to(root).as_posix(): p.stat().st_size
            for p in root.rglob("*")
            if p.is_file()
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _describe_files(cls, root: Path) -> Dict[str, Dict[str, Any]]:
        """Map of posix relative path -> {size, sha256} for every regular file under root."""
        return {
            rel: {"size": size, "sha256": cls._sha256_file(root / rel)}
            for rel, size in cls._list_files(root).items()
        }

    @classmethod
    def _mismatched_files(
        cls, root: Path, expected: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Compare files under root with the marker's list; empty lists mean a match."""
        actual = cls._list_files(root)
        size_mismatch, hash_mismatch = [], []
        for rel, exp in expected.items():
            if rel not in actual:
                continue
            if actual[rel] != exp["size"]:
                size_mismatch.append(rel)
            elif "sha256" in exp and cls._sha256_file(root / rel) != exp["sha256"]:
                hash_mismatch.append(rel)
        return {
            "missing": sorted(set(expected) - set(actual)),
            "unexpected": sorted(set(actual) - set(expected)),
            "size_mismatch": sorted(size_mismatch),
            "sha256_mismatch": sorted(hash_mismatch),
        }

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

        Files are written to a private staging dir and renamed into place, so
        readers never see a half-written entry.

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
        incoming = sum(self._list_files(source_dir).values()) if source_dir.exists() else 0
        self._make_room(incoming, protect=cache_key)

        with self._entry_lock(cache_key):
            staging = self._staging_dir(cache_key)
            try:
                staging_output = staging / "output"
                staging.mkdir(parents=True)
                # Reflink, else hard-link, else copy: no second full write of
                # the output when the filesystem allows it.
                if source_dir.exists():
                    clone_tree(source_dir, staging_output)
                else:
                    staging_output.mkdir(parents=True, exist_ok=True)

                files = self._list_files(staging_output)
                size = sum(files.values())

                manifest = {
                    "cache_key": cache_key,
                    "task_id": task_id,
                    "tool_name": tool_name,
                    "parent_hashes": parent_hashes or [],
                    "created_at": datetime.utcnow().isoformat(),
                    "size_bytes": size,
                    "files": files,
                    "metadata": metadata or {}
                }
                (staging / ARTIFACT_MANIFEST).write_text(json.dumps(manifest, indent=2))
                # Read-only so an in-place edit through a hard link fails
                # instead of silently changing the cached copy.
                make_tree_read_only(staging_output)
                (staging / ".success").touch()
                final = self._publish_local(staging, cache_key)
                self.lru.touch(cache_key)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        output_dir = final / "output"

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
        
        # Upload all files, then the manifest, then the completion marker
        # LAST: artifact_exists() only reports a hit once the marker is there,
        # so a crash mid-upload leaves a miss rather than a truncated hit.
        # Byte-identical copies of dataset files are left out (module docstring).
        prefix = self.minio.get_artifact_key(cache_key)
        files = self._describe_files(source_dir)
        from_dataset = self._dataset_copies(files)
        if from_dataset:
            count = self.minio.upload_directory(source_dir, prefix, exclude=set(from_dataset))
        else:
            count = self.minio.upload_directory(source_dir, prefix)
        if count != len(files) - len(from_dataset):
            logger.error(
                f"Uploaded {count}/{len(files) - len(from_dataset)} files for {cache_key[:12]}; "
                "not marking artifact complete"
            )
            return False
        for rel, content_id in from_dataset.items():
            files[rel].update(source=DATASET_SOURCE, dataset_content_id=content_id)

        manifest_path = source_dir.parent / ARTIFACT_MANIFEST
        if manifest_path.exists():
            if not self.minio.upload_file(
                manifest_path,
                f"{prefix}{ARTIFACT_MANIFEST}",
                content_type="application/json"
            ):
                logger.error(f"Failed to upload manifest for {cache_key[:12]}")
                return False

        marker = {
            "cache_key": cache_key,
            "files": files,
            "completed_at": datetime.utcnow().isoformat(),
        }
        if from_dataset:
            # Where each dataset is published, so a worker without it can sync it.
            marker["datasets"] = {
                cid: self._datasets[cid][2] for cid in set(from_dataset.values())
            }
        if not self.minio.upload_bytes(
            json.dumps(marker, sort_keys=True).encode(),
            f"{prefix}{ARTIFACT_COMPLETE_MARKER}",
            content_type="application/json",
        ):
            logger.error(f"Failed to upload completion marker for {cache_key[:12]}")
            return False

        if cache_key in self._artifacts:
            self._artifacts[cache_key].storage_type = "both"
            self._artifacts[cache_key].minio_key = prefix

        skipped = (
            f", {len(from_dataset)} identical to dataset {self._active_dataset[:12]} left out"
            if from_dataset else ""
        )
        logger.info(f"Uploaded artifact to MinIO: {cache_key[:12]} ({count} files{skipped})")
        return True

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
        
        prefix = self.minio.get_artifact_key(cache_key)
        marker = self._read_marker(prefix)
        expected = self._marker_files(marker)
        incoming = sum(e["size"] for e in expected.values()) if expected else 0
        self._make_room(incoming, protect=cache_key)

        with self._entry_lock(cache_key):
            if self.local_exists(cache_key):
                self.lru.touch(cache_key)
                return self.get_local_path(cache_key) / "output"

            staging = self._staging_dir(cache_key)
            staging_output = staging / "output"
            try:
                staging_output.mkdir(parents=True)
                count = 0
                for obj in self.minio.list_objects(prefix, recursive=True):
                    if obj.is_dir:
                        continue
                    rel = obj.object_name[len(prefix):].lstrip("/")
                    if not rel or rel == ARTIFACT_COMPLETE_MARKER:
                        continue
                    if rel == ARTIFACT_MANIFEST:
                        dest = staging / ARTIFACT_MANIFEST
                    else:
                        dest = staging_output / rel
                    if not self.minio.download_file(obj.object_name, dest):
                        raise RuntimeError(f"failed to download {obj.object_name}")
                    count += 1
                    # A tool output named manifest.json shares its object key
                    # with the cache manifest; keep a copy in output as well.
                    if rel == ARTIFACT_MANIFEST and expected and ARTIFACT_MANIFEST in expected:
                        shutil.copy2(dest, staging_output / ARTIFACT_MANIFEST)

                if expected is not None:
                    restored = self._restore_dataset_files(
                        staging_output, expected, self._marker_datasets(marker)
                    )
                    # Restored files were checked against the dataset's digests.
                    check = {
                        rel: ({"size": e["size"]} if rel in restored else e)
                        for rel, e in expected.items()
                    }
                    mismatch = self._mismatched_files(staging_output, check)
                    if any(mismatch.values()):
                        details = ", ".join(f"{k}={v[:5]}" for k, v in mismatch.items() if v)
                        raise RuntimeError(
                            f"downloaded files do not match completion marker ({details})"
                        )
                elif not self._list_files(staging_output):
                    raise RuntimeError("legacy artifact has no output files")

                make_tree_read_only(staging_output)
                (staging / ".success").touch()
                final = self._publish_local(staging, cache_key)
                self.lru.touch(cache_key)
            except Exception as e:
                shutil.rmtree(staging, ignore_errors=True)
                logger.warning(f"Download of artifact {cache_key[:12]} from MinIO failed: {e}")
                return None

        logger.info(f"Downloaded artifact from MinIO: {cache_key[:12]} ({count} files)")
        return final / "output"

    def _read_marker(self, prefix: str) -> Optional[Dict[str, Any]]:
        """The completion marker under prefix, or None (legacy entry or unreadable)."""
        marker_key = f"{prefix}{ARTIFACT_COMPLETE_MARKER}"
        if not self.minio.exists(marker_key):
            return None
        raw = self.minio.download_bytes(marker_key)
        if raw is None:
            return None
        try:
            marker = json.loads(raw)
        except ValueError:
            return None
        return marker if isinstance(marker, dict) else None

    @staticmethod
    def _marker_files(marker: Optional[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        File list from the completion marker, or None for legacy entries.

        Each entry is ``{"size": int, "sha256": str}``. Markers written before
        checksums were added store only the size, so ``sha256`` may be absent.
        Files left out as dataset copies also carry ``source`` and
        ``dataset_content_id``.
        """
        files = marker.get("files") if marker else None
        if not isinstance(files, dict):
            return None
        expected: Dict[str, Dict[str, Any]] = {}
        for rel, entry in files.items():
            if not isinstance(entry, dict):
                entry = {"size": entry}
            expected[str(rel)] = {"size": int(entry["size"])}
            if entry.get("sha256"):
                expected[str(rel)]["sha256"] = str(entry["sha256"])
            if entry.get("source") == DATASET_SOURCE:
                expected[str(rel)]["source"] = DATASET_SOURCE
                expected[str(rel)]["dataset_content_id"] = str(entry.get("dataset_content_id"))
        return expected

    @staticmethod
    def _marker_datasets(marker: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """content_id -> published location of the datasets the marker's files come from."""
        datasets = (marker or {}).get("datasets")
        if not isinstance(datasets, dict):
            return {}
        return {str(k): v for k, v in datasets.items() if isinstance(v, dict)}

    def _expected_files(self, prefix: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """File list from the completion marker under prefix (see _marker_files)."""
        return self._marker_files(self._read_marker(prefix))

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
        
        # Upload to MinIO if enabled (required for production cache path)
        if upload_to_remote and self.use_minio:
            if not self.minio or not self.minio.is_available:
                raise RuntimeError(
                    f"MinIO upload required for cache key {cache_key[:12]} but MinIO is unavailable"
                )
            if not self.upload_to_minio(cache_key):
                raise RuntimeError(
                    f"Failed to upload artifact {cache_key[:12]} to MinIO/GCS"
                )
            info.minio_key = self.minio.get_artifact_key(cache_key)
            if self.should_keep_local(cache_key):
                # Keep for local hits; LRU eviction frees it when space is needed.
                info.storage_type = "both"
            else:
                # Keep-local is off, or the entry alone exceeds the local cap.
                self.delete_local(cache_key)
                info.storage_type = "minio"
                info.local_path = None
        
        return info

    def delete_local(self, cache_key: str) -> bool:
        """Remove a cache entry from local disk (MinIO/GCS copy is kept)."""
        local_path = self.get_local_path(cache_key)
        if not local_path.exists():
            self._artifacts.pop(cache_key, None)
            return False
        with self._entry_lock(cache_key):
            shutil.rmtree(local_path, ignore_errors=True)
        self._artifacts.pop(cache_key, None)
        logger.debug(f"Dropped local cache entry: {cache_key[:12]}")
        return True
    
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
