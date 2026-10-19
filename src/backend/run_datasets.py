"""
Per-run datasets.

Every pipeline run uses the dataset and model script of its own config, so runs
on different datasets (e.g. a CIFAR-10 and a CelebA config) can share one
backend and the same workers. The start and restart routes prepare and publish
the run's dataset and record it here; ``GET /dataset?run_id=...`` (and
``/dataset/manifest``, ``/dataset/download-url``) serve it to workers.

A prepared dataset is reused by every later run with the same (name, variant,
params, poisoning). Preparation is serialized per dataset, so runs started at
the same time do not prepare or publish it twice.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.common import get_logger

logger = get_logger(__name__)


def dataset_key(
    name: str,
    variant: str,
    params: Optional[Dict[str, Any]] = None,
    poisoning: Optional[Dict[str, Any]] = None,
) -> str:
    """Registry key: everything that selects what prepare_dataset produces."""
    return json.dumps(
        {"name": name, "variant": variant, "params": params or {}, "poisoning": poisoning},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class RunDatasetRegistry:
    """
    Prepared datasets (by dataset_key) and each run's dataset info (by run id).

    A run's info is the shared dataset info plus that run's model script
    fields (model_script, model_script_sha256, model_script_minio_key).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prepare_locks: Dict[str, threading.Lock] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._runs: Dict[str, Dict[str, Any]] = {}

    def prepare_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._prepare_locks.setdefault(key, threading.Lock())

    def get_dataset(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            info = self._datasets.get(key)
            return dict(info) if info is not None else None

    def put_dataset(self, key: str, info: Dict[str, Any]) -> None:
        with self._lock:
            self._datasets[key] = dict(info)

    def set_run(self, run_id: str, info: Dict[str, Any]) -> None:
        with self._lock:
            self._runs[run_id] = dict(info)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            info = self._runs.get(run_id)
            return dict(info) if info is not None else None

    def run_ids(self) -> list:
        with self._lock:
            return list(self._runs)


def resolve_model_script(script: Optional[str], config_path: Optional[str] = None) -> Optional[Path]:
    """Model script file: absolute, else relative to the cwd, else to the config's dir."""
    if not script:
        return None
    path = Path(script)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]
    if not path.is_absolute() and config_path:
        candidates.append(Path(config_path).resolve().parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_dataset(
    registry: RunDatasetRegistry,
    manager,
    store,
    name: str,
    variant: str,
    params: Dict[str, Any],
    poisoning: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Prepared (and, with a store, published) dataset info; prepared once per key."""
    key = dataset_key(name, variant, params, poisoning)
    with registry.prepare_lock(key):
        info = registry.get_dataset(key)
        if info is not None:
            logger.info(f"Reusing prepared dataset {name}/{variant} ({(info.get('content_id') or '')[:12]})")
            return info

        ds_info = manager.prepare_dataset(name=name, variant=variant, poisoning=poisoning, **params)
        if not ds_info:
            raise RuntimeError(f"Dataset preparation returned no dataset info for {name}/{variant}")
        info = ds_info.to_dict()

        published = True
        if store is not None and getattr(store, "is_available", False):
            # The output dir name ("clean", "poisoned", ...) keeps variants apart in MinIO.
            minio_key = f"datasets/{name}/{Path(info['output_dir']).name}"
            try:
                from src.backend.dataset_publish import publish_dataset

                # Skips the upload when MinIO already has a manifest with the same content_id.
                if not publish_dataset(store, info, minio_key):
                    raise RuntimeError(f"dataset upload to {minio_key} incomplete")
            except Exception as e:
                published = False
                logger.warning(f"Failed to upload dataset to MinIO: {e}")
        # A failed publish is retried by the next run that needs this dataset.
        if published:
            registry.put_dataset(key, info)
        return info


def _attach_model_script(store, info: Dict[str, Any], model_path: Optional[Path], script: Optional[str]) -> None:
    """Add the run's model script fields to its dataset info (uploading it to MinIO)."""
    info["model_script"] = str(model_path) if model_path else script
    info["model_script_sha256"] = None
    info["model_script_minio_key"] = None
    if model_path is None:
        if script:
            logger.warning(f"Model script path not found: {script}")
        return
    sha = _sha256(model_path)
    info["model_script_sha256"] = sha
    if store is None or not getattr(store, "is_available", False):
        return
    # Content-addressed: runs on the same dataset with different model scripts
    # get different keys, and workers can trust a local copy with the same hash.
    out_name = Path(info.get("output_dir") or info.get("variant") or "clean").name
    key = f"datasets/{info.get('name')}/{out_name}/models/{sha[:16]}/config_model.py"
    try:
        if store.exists(key) or store.upload_file(model_path, key):
            info["model_script_minio_key"] = key
            logger.info(f"Model script available in MinIO: {key}")
        else:
            logger.warning(f"Failed to upload model script to MinIO: {model_path}")
    except Exception as e:
        logger.warning(f"Failed to upload model script to MinIO: {e}")


def prepare_run_dataset(
    registry: RunDatasetRegistry,
    run_id: str,
    dataset: Dict[str, Any],
    model_script: Optional[str],
    manager,
    store=None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare and publish the dataset of one run and record it under ``run_id``.

    ``dataset`` is the run's effective dataset config ({"name", "variant",
    "params"}, overrides applied), ``model_script`` its model script. Returns
    the run's dataset info.
    """
    name = dataset.get("name")
    if not name:
        raise ValueError(f"Run {run_id} has no dataset name")
    variant = dataset.get("variant") or "clean"
    params = dict(dataset.get("params") or {})
    poisoning = params.pop("poisoning", None)
    if variant != "poisoned":
        poisoning = None

    info = _ensure_dataset(registry, manager, store, name, variant, params, poisoning)
    info["run_id"] = run_id
    info["config"] = dict(dataset)
    _attach_model_script(store, info, resolve_model_script(model_script, config_path), model_script)
    registry.set_run(run_id, info)
    logger.info(
        f"Run {run_id} uses dataset {name}/{variant} "
        f"({(info.get('content_id') or 'no content_id')[:12]}), model script {info.get('model_script')}"
    )
    return info
