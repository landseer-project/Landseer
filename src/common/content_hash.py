"""
Content hashes for datasets.

Cache keys and lineage checks must identify a dataset by what is in it, not
by where it lives or when it was copied, so the same files give the same hash
on every host.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .pylogger import get_logger

logger = get_logger(__name__)

# Only the arrays define the dataset; config_model.py and other files that
# live in the same directory are ignored.
DATASET_FILES = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")

# Per-directory cache of file digests keyed by size and mtime, so large
# arrays are hashed once rather than on every task.
HASH_SIDECAR = ".landseer_hashes.json"

_CHUNK = 8 * 1024 * 1024


def file_sha256(path: Union[str, Path]) -> str:
    """sha256 of a file's contents, read in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sidecar(path: Path) -> Dict[str, dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sidecar(path: Path, data: Dict[str, dict]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except OSError as e:
        # Read-only dataset dirs are fine; we just rehash next time.
        logger.debug(f"Could not write dataset hash sidecar {path}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass


def dataset_content_hash(data_dir: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Content hash of a dataset directory.

    sha256 over the sorted (file name, file sha256) pairs of the DATASET_FILES
    present in ``data_dir``. Returns None if the directory is missing or holds
    none of those files.
    """
    if not data_dir:
        return None
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None

    sidecar_path = data_dir / HASH_SIDECAR
    sidecar = _read_sidecar(sidecar_path)
    digests: Dict[str, str] = {}
    changed = False

    for name in DATASET_FILES:
        p = data_dir / name
        try:
            st = p.stat()
        except OSError:
            continue
        if not p.is_file():
            continue
        cached = sidecar.get(name)
        if (
            isinstance(cached, dict)
            and cached.get("size") == st.st_size
            and cached.get("mtime_ns") == st.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            digests[name] = cached["sha256"]
            continue
        digest = file_sha256(p)
        digests[name] = digest
        sidecar[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        changed = True

    if not digests:
        return None
    if changed:
        _write_sidecar(sidecar_path, sidecar)

    payload = json.dumps(sorted(digests.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
