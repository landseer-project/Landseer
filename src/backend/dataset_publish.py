"""
Publish a prepared dataset to MinIO with its manifest.

The manifest's files are uploaded first and ``dataset_manifest.json`` last, so
a manifest in MinIO means the dataset upload is complete. Workers download
exactly the files the manifest lists and verify them against it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from src.common import get_logger
from src.common import dataset_manifest as dsm

logger = get_logger(__name__)

UPLOAD_WORKERS = 8


def publish_dataset(store, dataset_info: Dict[str, Any], dataset_key: str) -> bool:
    """
    Upload the dataset in ``dataset_info["output_dir"]`` under ``dataset_key``.

    Skips the upload when MinIO already holds a manifest with the same
    content_id. On success sets ``minio_key`` and ``manifest_minio_key`` in
    ``dataset_info`` and returns True.
    """
    data_dir = Path(dataset_info["output_dir"])
    manifest = dsm.load_manifest(data_dir)
    manifest_key = f"{dataset_key}/{dsm.MANIFEST_NAME}"

    if manifest is None:
        # Should not happen (prepare_dataset writes one); keep the old behaviour.
        logger.warning(f"No dataset manifest in {data_dir}; uploading directory without one")
        store.upload_directory(data_dir, dataset_key)
        dataset_info["minio_key"] = dataset_key
        return True

    remote = dsm.parse_manifest(store.download_bytes(manifest_key)) if store.exists(manifest_key) else None
    if remote and remote.get("content_id") == manifest["content_id"]:
        logger.info(f"Dataset already in MinIO with content_id {manifest['content_id'][:12]}: {dataset_key}")
        dataset_info["minio_key"] = dataset_key
        dataset_info["manifest_minio_key"] = manifest_key
        return True
    if remote:
        # Withdraw the old completion marker before replacing any files.
        store.delete(manifest_key)

    rels = list(manifest["files"])
    logger.info(
        f"Uploading dataset {dataset_key}: {len(rels)} files, "
        f"{manifest.get('total_bytes', 0) / 1024 ** 3:.1f} GB"
    )

    def _upload(rel: str) -> bool:
        return store.upload_file(data_dir / rel, f"{dataset_key}/{rel}")

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        results = list(pool.map(_upload, rels))
    failed = [rel for rel, ok in zip(rels, results) if not ok]
    if failed:
        logger.error(f"Dataset upload incomplete ({len(failed)} files failed, e.g. {failed[:3]}); no manifest uploaded")
        return False

    if not store.upload_file(data_dir / dsm.MANIFEST_NAME, manifest_key, content_type="application/json"):
        logger.error(f"Failed to upload dataset manifest {manifest_key}")
        return False

    dataset_info["minio_key"] = dataset_key
    dataset_info["manifest_minio_key"] = manifest_key
    logger.info(f"Dataset published to MinIO: {dataset_key} (content_id {manifest['content_id'][:12]})")
    return True
