"""
Dataset manifests: content identity plus provenance for a prepared dataset.

The backend writes ``dataset_manifest.json`` into the dataset directory when it
prepares (or first uses) a dataset, and uploads it to MinIO after the files,
so its presence there also marks the upload complete. Layout::

    {
      "schema_version": 1,
      "content_id": "<sha256>",          # dataset identity (cache keys, lineage)
      "file_count": 7, "total_bytes": 9000000000,
      "files": {"data.npy": {"size": ..., "sha256": ...}, ...},
      "selection": {"required_files": [...], "required_dirs": [...], "ignored": [...]},
      "provenance": {..., "provenance_id": "<sha256>"}   # how it was made; NOT in content_id
    }

``content_id`` depends only on (relative path, size, sha256) of the selected
files, so the same files give the same id on every host regardless of path or
mtime. Provenance is recorded alongside but never changes the id.

Hashes are cached per file in ``.landseer_hashes.json`` (keyed by relative
path, trusted only while size and mtime match), so re-checking a 180k-file
dataset is a stat walk, and new files are hashed in parallel.
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .pylogger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "dataset_manifest.json"
HASH_SIDECAR = ".landseer_hashes.json"
# Files that can live in a dataset directory but are not part of the dataset.
IGNORED_NAMES = frozenset({"config_model.py", "dataset_meta.json", MANIFEST_NAME, HASH_SIDECAR})
# Provenance fields that describe the run, not the recipe; excluded from provenance_id.
NON_DETERMINISTIC_PROVENANCE = frozenset({"prep_host", "prepared_at", "manifest_created_at", "provenance_id"})

_CHUNK = 8 * 1024 * 1024
PathLike = Union[str, Path]


def file_sha256(path: PathLike) -> str:
    """sha256 of a file's contents, read in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2)))


def _is_ignored(rel: str) -> bool:
    parts = rel.split("/")
    return any(p.startswith(".") for p in parts) or parts[-1] in IGNORED_NAMES


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def _walk(root: Path, base: Path) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in filenames:
            full = Path(dirpath) / name
            if full.is_file():
                yield full.relative_to(base).as_posix()


def select_files(
    data_dir: PathLike,
    required_files: Optional[Iterable[str]] = None,
    required_dirs: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Relative paths that make up the dataset, sorted.

    With neither list given, every file under ``data_dir`` is selected. Hidden
    paths and IGNORED_NAMES are always skipped. Listed files that do not exist
    are left out (the caller decides whether that is an error).
    """
    root = Path(data_dir)
    selected = set()
    if required_files is None and required_dirs is None:
        selected.update(_walk(root, root))
    else:
        for name in required_files or ():
            if (root / name).is_file():
                selected.add(Path(name).as_posix())
        for name in required_dirs or ():
            if (root / name).is_dir():
                selected.update(_walk(root / name, root))
    return sorted(rel for rel in selected if not _is_ignored(rel))


def list_unlisted_files(data_dir: PathLike, manifest: Dict[str, Any]) -> List[str]:
    """Files present under data_dir (not ignored) that the manifest does not list."""
    listed = set(manifest.get("files", {}))
    return sorted(rel for rel in _walk(Path(data_dir), Path(data_dir))
                  if rel not in listed and not _is_ignored(rel))


# ---------------------------------------------------------------------------
# Hashing with a per-directory sidecar
# ---------------------------------------------------------------------------

def _read_sidecar(path: Path) -> Dict[str, dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sidecar(path: Path, data: Dict[str, dict]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True))
        os.replace(tmp, path)
    except OSError as e:
        logger.debug(f"Could not write hash sidecar {path}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass


def hash_files(
    data_dir: PathLike,
    relpaths: Iterable[str],
    workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    ``{relpath: {"size", "sha256"}}`` for the given files (missing ones are
    omitted). Digests are reused from the sidecar when size and mtime match;
    the rest are hashed in a thread pool and written back to the sidecar.
    """
    root = Path(data_dir)
    sidecar_path = root / HASH_SIDECAR
    sidecar = _read_sidecar(sidecar_path)
    result: Dict[str, Dict[str, Any]] = {}
    todo: List[tuple] = []

    for rel in relpaths:
        try:
            st = (root / rel).stat()
        except OSError:
            continue
        cached = sidecar.get(rel)
        if (
            isinstance(cached, dict)
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            result[rel] = {"size": st.st_size, "sha256": cached["sha256"]}
        else:
            todo.append((rel, st.st_size, st.st_mtime_ns))

    if todo:
        total = sum(size for _, size, _ in todo)
        if total > 1024 ** 3 or len(todo) > 1000:
            logger.info(f"Hashing {len(todo)} dataset files ({total / 1024 ** 3:.1f} GB) in {root}")
        with ThreadPoolExecutor(max_workers=workers or _default_workers()) as pool:
            digests = list(pool.map(lambda item: file_sha256(root / item[0]), todo))
        for (rel, size, mtime_ns), digest in zip(todo, digests):
            result[rel] = {"size": size, "sha256": digest}
            sidecar[rel] = {"size": size, "mtime_ns": mtime_ns, "sha256": digest}
        _write_sidecar(sidecar_path, sidecar)

    return result


def compute_content_id(files: Dict[str, Dict[str, Any]]) -> str:
    """sha256 over the sorted (relpath, size, sha256) triples."""
    triples = sorted([rel, int(e["size"]), str(e["sha256"])] for rel, e in files.items())
    payload = json.dumps(triples, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def provenance_id(provenance: Dict[str, Any]) -> str:
    """Hash of the deterministic provenance fields (host and timestamps excluded)."""
    stable = {k: v for k, v in provenance.items() if k not in NON_DETERMINISTIC_PROVENANCE}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def with_provenance_id(provenance: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(provenance)
    out["provenance_id"] = provenance_id(out)
    return out


# ---------------------------------------------------------------------------
# Manifest build / IO / verification
# ---------------------------------------------------------------------------

def build_manifest(
    data_dir: PathLike,
    required_files: Optional[Iterable[str]] = None,
    required_dirs: Optional[Iterable[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Hash the selected files of ``data_dir`` and return a manifest dict."""
    req_files = list(required_files) if required_files is not None else None
    req_dirs = list(required_dirs) if required_dirs is not None else None
    rels = select_files(data_dir, req_files, req_dirs)
    files = hash_files(data_dir, rels, workers=workers)
    return {
        "schema_version": SCHEMA_VERSION,
        "content_id": compute_content_id(files),
        "file_count": len(files),
        "total_bytes": sum(e["size"] for e in files.values()),
        "files": dict(sorted(files.items())),
        "selection": {
            "required_files": req_files,
            "required_dirs": req_dirs,
            "ignored": sorted(IGNORED_NAMES) + ["hidden files"],
        },
        "provenance": with_provenance_id(provenance or {}),
    }


def write_manifest(data_dir: PathLike, manifest: Dict[str, Any]) -> Path:
    path = Path(data_dir) / MANIFEST_NAME
    tmp = path.with_name(f".{MANIFEST_NAME}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return path


def parse_manifest(raw: Union[str, bytes, None]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    try:
        manifest = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return None
    if not manifest.get("content_id"):
        return None
    return manifest


def load_manifest(data_dir: PathLike) -> Optional[Dict[str, Any]]:
    try:
        return parse_manifest((Path(data_dir) / MANIFEST_NAME).read_text())
    except OSError:
        return None


def summarize(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The small part of a manifest that travels in dataset info / API responses."""
    prov = manifest.get("provenance") or {}
    return {
        "content_id": manifest.get("content_id"),
        "provenance_id": prov.get("provenance_id"),
        "provenance": prov,
        "file_count": manifest.get("file_count"),
        "total_bytes": manifest.get("total_bytes"),
        "schema_version": manifest.get("schema_version"),
    }


@dataclass
class VerifyResult:
    content_id: Optional[str] = None  # id of the local copy over the manifest's file list
    missing: List[str] = field(default_factory=list)
    size_mismatch: List[str] = field(default_factory=list)
    hash_mismatch: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.size_mismatch or self.hash_mismatch)

    def describe(self) -> str:
        parts = [f"{name}={vals[:5]}" for name, vals in (
            ("missing", self.missing),
            ("size_mismatch", self.size_mismatch),
            ("sha256_mismatch", self.hash_mismatch),
        ) if vals]
        return ", ".join(parts) or "ok"


def verify_dir(
    data_dir: PathLike,
    manifest: Dict[str, Any],
    workers: Optional[int] = None,
) -> VerifyResult:
    """Check data_dir's copy of every manifest file (size first, then sha256)."""
    expected: Dict[str, Dict[str, Any]] = manifest.get("files", {})
    root = Path(data_dir)
    res = VerifyResult()
    to_hash = []
    for rel, exp in expected.items():
        try:
            size = (root / rel).stat().st_size
        except OSError:
            res.missing.append(rel)
            continue
        if size != int(exp["size"]):
            res.size_mismatch.append(rel)
        else:
            to_hash.append(rel)
    actual = hash_files(root, to_hash, workers=workers)
    for rel in to_hash:
        if actual.get(rel, {}).get("sha256") != expected[rel]["sha256"]:
            res.hash_mismatch.append(rel)
    res.missing.sort()
    res.size_mismatch.sort()
    res.hash_mismatch.sort()
    if res.ok:
        res.content_id = manifest.get("content_id")
    else:
        local = {rel: actual[rel] for rel in actual}
        for rel in res.size_mismatch:
            try:
                local.update(hash_files(root, [rel], workers=1))
            except OSError:
                pass
        res.content_id = compute_content_id(local) if local else None
    return res
