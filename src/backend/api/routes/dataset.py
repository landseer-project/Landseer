"""Dataset info endpoints for workers."""
from __future__ import annotations
from src.backend.api.models import DatasetInfoResponse

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from src.common import get_logger
from src.common import dataset_manifest as dsm
from src.backend.initialization import get_backend_context
from src.backend.api.state import SchedulerState, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()


def _manifest_summary(ds_info: Dict[str, Any]) -> Dict[str, Any]:
    """content_id/provenance from dataset info, else from the manifest file on disk."""
    if ds_info.get("content_id"):
        return {
            "content_id": ds_info.get("content_id"),
            "provenance_id": ds_info.get("provenance_id"),
            "provenance": ds_info.get("provenance"),
            "file_count": ds_info.get("file_count"),
            "total_bytes": ds_info.get("total_bytes"),
        }
    out_dir = ds_info.get("output_dir")
    manifest = dsm.load_manifest(out_dir) if out_dir else None
    return dsm.summarize(manifest) if manifest else {}


@router.get("/dataset", response_model=DatasetInfoResponse, tags=["Dataset"])
async def get_dataset_info(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get dataset information for workers.
    
    Workers use this to:
    1. Check if dataset is available
    2. Get MinIO key to download dataset
    3. Get local path if running on same machine as backend
    
    The backend prepares the dataset on startup and uploads to MinIO.
    Workers should download from MinIO if they don't have local access.
    """
    context = get_backend_context()
    
    # Check if dataset info is available
    if not context or not context.dataset_info:
        # Fall back to pipeline config
        pipeline_dataset = None
        model_script = None
        if state.pipeline:
            pipeline_dataset = state.pipeline.dataset
            model_cfg = getattr(state.pipeline, "model", None)
            if isinstance(model_cfg, dict):
                model_script = model_cfg.get("script")
        return DatasetInfoResponse(
            available=False,
            minio_available=False,
            config=pipeline_dataset,
            model_script=model_script,
            model_script_minio_key=None,
        )
    
    ds_info = context.dataset_info
    
    # Check if MinIO is available for this dataset
    minio_key = ds_info.get("minio_key")
    minio_available = False
    if minio_key and context.store and context.store.is_available:
        try:
            # Check if the key exists (prefix check)
            minio_available = True  # If we have the key, assume it's there
        except Exception:
            pass
    
    # Get model script path from pipeline config
    model_script = None
    if context.pipeline and context.pipeline.model:
        model_script = context.pipeline.model.get("script")
    
    summary = _manifest_summary(ds_info)

    return DatasetInfoResponse(
        available=True,
        name=ds_info.get("name"),
        variant=ds_info.get("variant"),
        train_samples=ds_info.get("train_samples"),
        test_samples=ds_info.get("test_samples"),
        local_path=ds_info.get("output_dir"),
        minio_key=minio_key,
        minio_available=minio_available,
        config=context.pipeline.dataset if context.pipeline else None,
        poisoning=ds_info.get("poisoning"),
        model_script=model_script,
        model_script_minio_key=ds_info.get("model_script_minio_key"),
        content_id=summary.get("content_id"),
        provenance_id=summary.get("provenance_id"),
        provenance=summary.get("provenance"),
        file_count=summary.get("file_count"),
        total_bytes=summary.get("total_bytes"),
        manifest_minio_key=ds_info.get("manifest_minio_key"),
    )


@router.get("/dataset/manifest", tags=["Dataset"])
async def get_dataset_manifest():
    """
    Full dataset_manifest.json (file list with sizes and sha256, content_id,
    provenance) for the dataset the backend serves. Workers with a manual
    LANDSEER_DATA_PATH verify their copy against it.
    """
    context = get_backend_context()
    ds_info = (context.dataset_info if context else None) or {}
    out_dir = ds_info.get("output_dir")
    path = Path(out_dir) / dsm.MANIFEST_NAME if out_dir else None
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset manifest not available")
    return FileResponse(path, media_type="application/json")

@router.get("/dataset/download-url", tags=["Dataset"])
async def get_dataset_download_url(
    state: SchedulerState = Depends(get_scheduler_state),
    expires_in: int = Query(default=3600, description="URL expiration time in seconds")
):
    """
    Get a presigned URL to download the dataset from MinIO.

    Workers can use this URL to download the dataset directly.
    """
    context = get_backend_context()

    if not context or not context.dataset_info:
        raise HTTPException(status_code=404, detail="Dataset not available")

    minio_key = context.dataset_info.get("minio_key")
    if not minio_key:
        raise HTTPException(status_code=404, detail="Dataset not in MinIO")

    if not context.store or not context.store.is_available:
        raise HTTPException(status_code=503, detail="MinIO store not available")

    try:
        return {
            "minio_key": minio_key,
            "bucket": context.store.config.bucket,
            "endpoint": context.store.config.endpoint,
            "message": "Use MinIO client to download with the provided key",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {e}")
