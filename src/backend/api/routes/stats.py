"""Database and system stats endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.common import get_logger
from src.backend.initialization import get_backend_context
from src.backend.api.state import SchedulerState, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()

@router.get("/stats/database", tags=["Statistics"])
async def get_database_stats(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get database statistics.
    
    Returns information about database connectivity and stored data.
    """
    if not state.db_service or not state.db_service.is_available():
        return {
            "available": False,
            "message": "Database service not available"
        }
    
    # Get evaluation result counts
    evaluation_count = 0
    try:
        from src.db.models import EvaluationResultModel
        from src.db import session_scope
        
        with session_scope() as session:
            evaluation_count = session.query(EvaluationResultModel).count()
    except Exception as e:
        logger.warning(f"Failed to get evaluation result count: {e}")
    
    stats = {
        "available": True,
        "task_progress": state.db_service.get_task_progress(),
        "worker_stats": state.db_service.get_worker_stats(),
        "evaluation_results": {
            "total_count": evaluation_count,
            "message": f"Found {evaluation_count} evaluation results in database"
        }
    }
    
    # Add pipeline-specific evaluation count if scheduler is initialized
    if state.scheduler and state.scheduler.pipeline:
        try:
            from src.db.models import EvaluationResultModel
            from src.db import session_scope
            
            with session_scope() as session:
                pipeline_eval_count = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == state.scheduler.pipeline.id
                ).count()
                stats["evaluation_results"]["pipeline_count"] = pipeline_eval_count
                stats["evaluation_results"]["pipeline_id"] = state.scheduler.pipeline.id
        except Exception as e:
            logger.warning(f"Failed to get pipeline evaluation count: {e}")
    
    return stats

@router.get("/stats/store", tags=["Statistics"])
async def get_store_stats():
    """
    Get artifact store (MinIO) statistics.
    
    Returns information about MinIO connectivity and stored artifacts.
    """
    context = get_backend_context()
    
    if not context or not context.store:
        return {
            "available": False,
            "message": "Artifact store not available"
        }
    
    store = context.store
    if not store.is_available:
        return {
            "available": False,
            "message": "MinIO connection not available"
        }
    
    # Get artifact count and size
    artifact_count = 0
    total_size = 0
    seen_keys = set()
    
    for obj in store.list_objects("artifacts/"):
        total_size += obj.size or 0
        parts = obj.object_name.split("/")
        if len(parts) >= 2:
            cache_key = parts[1]
            if cache_key not in seen_keys:
                seen_keys.add(cache_key)
                artifact_count += 1
    
    return {
        "available": True,
        "endpoint": store.config.endpoint,
        "bucket": store.config.bucket,
        "artifact_count": artifact_count,
        "total_size_bytes": total_size,
        "total_size_human": _format_size(total_size)
    }

def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

@router.get("/stats/system", tags=["Statistics"])
async def get_system_stats(state: SchedulerState = Depends(get_scheduler_state)):
    """
    Get overall system statistics.
    
    Combines database, store, and scheduler statistics.
    """
    context = get_backend_context()
    
    # Database status
    db_available = state.db_service and state.db_service.is_available()
    
    # Store status
    store_available = context and context.store and context.store.is_available
    
    # Scheduler status
    scheduler_active = state.is_initialized()
    
    return {
        "scheduler_active": scheduler_active,
        "database_available": db_available,
        "store_available": store_available,
        "workers_registered": len(state.workers),
        "tasks_tracked": len(state.task_metadata),
        "started_at": state.started_at.isoformat() if state.started_at else None
    }
