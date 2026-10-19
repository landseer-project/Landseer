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


def _run_dataset_info(state: SchedulerState, run_id: str) -> Dict[str, Any]:
    """Dataset info the backend recorded for run_id (404 if it has none)."""
    info = state.run_datasets.get_run(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No dataset recorded for run '{run_id}'")
    return info


def _minio_available(context, ds_info: Dict[str, Any]) -> bool:
    # If we have the key, assume it's there (published before it is recorded).
    return bool(ds_info.get("minio_key") and context and context.store and context.store.is_available)


def _dataset_response(context, ds_info: Dict[str, Any], config, model_script) -> DatasetInfoResponse:
    summary = _manifest_summary(ds_info)
    return DatasetInfoResponse(
        available=True,
        run_id=ds_info.get("run_id"),
        name=ds_info.get("name"),
        variant=ds_info.get("variant"),
        train_samples=ds_info.get("train_samples"),
        test_samples=ds_info.get("test_samples"),
        local_path=ds_info.get("output_dir"),
        minio_key=ds_info.get("minio_key"),
        minio_available=_minio_available(context, ds_info),
        config=config,
        poisoning=ds_info.get("poisoning"),
        model_script=model_script,
        model_script_minio_key=ds_info.get("model_script_minio_key"),
        model_script_sha256=ds_info.get("model_script_sha256"),
        content_id=summary.get("content_id"),
        provenance_id=summary.get("provenance_id"),
        provenance=summary.get("provenance"),
        file_count=summary.get("file_count"),
        total_bytes=summary.get("total_bytes"),
        manifest_minio_key=ds_info.get("manifest_minio_key"),
    )


@router.get("/dataset", response_model=DatasetInfoResponse, tags=["Dataset"])
async def get_dataset_info(
    state: SchedulerState = Depends(get_scheduler_state),
    run_id: Optional[str] = Query(default=None, description="Pipeline run whose dataset to return"),
):
    """
    Get dataset information for workers.
    
    Workers use this to:
    1. Check if dataset is available
    2. Get MinIO key to download dataset
    3. Get local path if running on same machine as backend
    
    With ``run_id``: the dataset and model script of that run (404 if the
    backend recorded none). Without: the most recently prepared dataset
    (legacy single-dataset view).
    Workers should download from MinIO if they don't have local access.
    """
    context = get_backend_context()

    if run_id is not None:
        ds_info = _run_dataset_info(state, run_id)
        return _dataset_response(context, ds_info, ds_info.get("config"), ds_info.get("model_script"))
    
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
    
    # Get model script path from pipeline config
    model_script = None
    if context.pipeline and context.pipeline.model:
        model_script = context.pipeline.model.get("script")
    
    return _dataset_response(
        context, ds_info, context.pipeline.dataset if context.pipeline else None, model_script
    )


@router.get("/dataset/manifest", tags=["Dataset"])
async def get_dataset_manifest(
    state: SchedulerState = Depends(get_scheduler_state),
    run_id: Optional[str] = Query(default=None, description="Pipeline run whose dataset manifest to return"),
):
    """
    Full dataset_manifest.json (file list with sizes and sha256, content_id,
    provenance) for the dataset the backend serves (for ``run_id``: that run's
    dataset). Workers with a manual data path verify their copy against it.
    """
    if run_id is not None:
        ds_info = _run_dataset_info(state, run_id)
    else:
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
    expires_in: int = Query(default=3600, description="URL expiration time in seconds"),
    run_id: Optional[str] = Query(default=None, description="Pipeline run whose dataset to return"),
):
    """
    Get a presigned URL to download the dataset from MinIO.

    Workers can use this URL to download the dataset directly.
    """
    context = get_backend_context()

    if run_id is not None:
        ds_info = _run_dataset_info(state, run_id)
    elif not context or not context.dataset_info:
        raise HTTPException(status_code=404, detail="Dataset not available")
    else:
        ds_info = context.dataset_info

    minio_key = ds_info.get("minio_key")
    if not minio_key:
        raise HTTPException(status_code=404, detail="Dataset not in MinIO")

    if not context or not context.store or not context.store.is_available:
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
