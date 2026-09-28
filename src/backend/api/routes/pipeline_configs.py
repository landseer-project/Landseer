"""Pipeline config and run management endpoints."""
from __future__ import annotations

import asyncio
import shutil
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.backend.initialization import get_backend_context, set_backend_context
from src.backend.scheduler import Scheduler
from src.backend.api.models import (
    PipelineConfigResponse,
    PipelineConfigListResponse,
    StartPipelineRunRequest,
    PipelineRunResponse,
    PipelineRunListResponse,
    RestartPipelineRunRequest,
    PipelineMetricsResponse,
    WorkflowMetrics,
)
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state
from src.backend.api.helpers import _require_pipeline_key

logger = get_logger(__name__)
router = APIRouter()

@router.get("/api/pipeline-configs", response_model=PipelineConfigListResponse, tags=["Pipeline Configs"])
async def get_pipeline_configs():
    """Get all available pipeline configurations."""
    try:
        from src.backend.config_discovery import sync_configs_to_db, get_all_configs
        
        # Sync configs from filesystem to DB
        sync_configs_to_db()
        
        # Get all configs
        configs = get_all_configs()
        
        return PipelineConfigListResponse(
            configs=[
                PipelineConfigResponse(
                    id=config.id,
                    name=config.name,
                    description=config.description,
                    config_path=config.config_path,
                    attack_config_path=config.attack_config_path,
                    config_hash=config.config_hash,
                    created_at=config.created_at.isoformat() if config.created_at else None,
                    updated_at=config.updated_at.isoformat() if config.updated_at else None,
                )
                for config in configs
            ],
            total=len(configs)
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline configs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline configs: {str(e)}")

@router.get("/api/pipeline-configs/{config_id}", response_model=PipelineConfigResponse, tags=["Pipeline Configs"])
async def get_pipeline_config(config_id: str):
    """Get a specific pipeline configuration."""
    try:
        from src.backend.config_discovery import get_config_by_id
        
        config = get_config_by_id(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Pipeline config '{config_id}' not found")
        
        return PipelineConfigResponse(
            id=config.id,
            name=config.name,
            description=config.description,
            config_path=config.config_path,
            attack_config_path=config.attack_config_path,
            config_hash=config.config_hash,
            created_at=config.created_at.isoformat() if config.created_at else None,
            updated_at=config.updated_at.isoformat() if config.updated_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline config {config_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline config: {str(e)}")

@router.get("/api/model-configs", tags=["Pipeline Configs"])
async def list_model_configs():
    """List available model configuration scripts."""
    import glob as glob_mod
    scripts = sorted(glob_mod.glob("configs/model/*.py"))
    return {
        "models": [
            {
                "path": s,
                "name": Path(s).stem,
            }
            for s in scripts
        ],
        "total": len(scripts),
    }

@router.post("/api/pipeline-configs/{config_id}/runs", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def start_pipeline_run(
    config_id: str,
    request: StartPipelineRunRequest,
    state: SchedulerState = Depends(get_scheduler_state),
    _auth: None = Depends(_require_pipeline_key),
):
    """Start a new pipeline run for a configuration."""

    def _start_pipeline_run_background(
        run_id: str,
        config_path: str,
        config_name: str,
        run_number: int,
        request_data: Dict[str, Any],
        state: SchedulerState,
    ) -> None:
        """
        Build pipeline + initialize scheduler in background.

        This keeps the HTTP request fast for large configs (e.g., trades) while
        preserving run state transitions in DB.
        """
        from src.db import session_scope, PipelineRunRepository, PipelineRunStatus
        from src.pipeline.config_loader import create_pipeline_from_config, load_pipeline_config
        from src.data import DatasetManager

        try:
            # Step 1b: transition to RUNNING immediately after background init starts.
            # Pipeline creation for large configs can take minutes; keeping PENDING
            # that whole time makes the UI look stuck even though work is active.
            with session_scope() as session:
                run_repo = PipelineRunRepository(session)
                run_repo.update_status(run_id, PipelineRunStatus.RUNNING)
            # Step 2: Create pipeline (CPU-heavy for large combination counts)
            t_create_start = time.time()
            pipeline = create_pipeline_from_config(
                config_path=config_path,
                tools_yaml_path="configs/tools.yaml",
                evaluators_yaml_path="configs/evaluators.yaml",
                pipeline_name=f"{config_name} (Run {run_number})",
                clear_registry=True,
                include_evaluation=True,
                dataset_name=request_data.get("dataset_name") or None,
                dataset_variant=request_data.get("dataset_variant") or None,
                tools_override=request_data.get("tools_override") or None,
                model_script=request_data.get("model_script") or None,
                attack_config_path=(
                    request_data.get("attack_config_path")
                    or request_data.get("config_attack_config_path")
                    or None
                ),
            )

            # Bind pipeline and all tasks/workflows to this run's ID
            pipeline.id = run_id
            for workflow in pipeline.workflows:
                workflow.pipeline_id = run_id
                workflow.run_id = run_id
                for task in workflow.tasks:
                    task.pipeline_id = run_id
                    task.run_id = run_id
                    # Propagate per-run cache policy to workers via task payload.
                    task.config["_run_use_cache"] = bool(request_data.get("use_cache", True))

            # Step 3: Dataset context (best-effort)
            ctx = get_backend_context()
            if ctx:
                ctx.pipeline = pipeline
                try:
                    cfg = load_pipeline_config(config_path)
                    effective_ds_name = request_data.get("dataset_name") or cfg.dataset.name
                    effective_ds_variant = request_data.get("dataset_variant") or cfg.dataset.variant
                    current_ds = ctx.dataset_info or {}
                    needs_prepare = (
                        not current_ds
                        or current_ds.get("name") != effective_ds_name
                        or current_ds.get("variant") != effective_ds_variant
                    )
                    if needs_prepare:
                        base_dir = Path("./data").resolve()
                        manager = ctx.dataset_manager or DatasetManager(base_dir)
                        poisoning = None
                        if effective_ds_variant == "poisoned":
                            poisoning = cfg.dataset.params.get("poisoning")
                        cfg.dataset.name = effective_ds_name
                        cfg.dataset.variant = effective_ds_variant
                        t_prepare_start = time.time()
                        ds_info = manager.prepare_dataset(
                            name=cfg.dataset.name,
                            variant=cfg.dataset.variant,
                            poisoning=poisoning,
                            **cfg.dataset.params,
                        )
                        if ds_info:
                            ctx.dataset_info = ds_info.to_dict()
                            ctx.dataset_manager = manager
                            if ctx.store and ctx.store.is_available:
                                dir_suffix = Path(ds_info.output_dir).name
                                dataset_key = f"datasets/{cfg.dataset.name}/{dir_suffix}"
                                try:
                                    marker = f"{dataset_key}/data.npy"
                                    if ctx.store.exists(marker):
                                        ctx.dataset_info["minio_key"] = dataset_key
                                        logger.info(
                                            f"Dataset already in MinIO ({marker}), skipping upload"
                                        )
                                    else:
                                        t_upload_start = time.time()
                                        ctx.store.upload_directory(ds_info.output_dir, dataset_key)
                                        ctx.dataset_info["minio_key"] = dataset_key
                                        logger.info(
                                            f"Dataset uploaded to MinIO in {time.time() - t_upload_start:.1f}s: {dataset_key}"
                                        )

                                    model_script = cfg.model.get("script") if cfg and cfg.model else None
                                    if model_script:
                                        model_path = Path(model_script)
                                        if not model_path.is_absolute():
                                            cfg_base = (
                                                Path(ctx.pipeline_config_path).resolve().parent
                                                if ctx.pipeline_config_path
                                                else Path(config_path).resolve().parent
                                            )
                                            model_path = (cfg_base / model_path).resolve()
                                        if model_path.exists() and model_path.is_file():
                                            model_script_key = f"{dataset_key}/config_model.py"
                                            if ctx.store.exists(model_script_key) or ctx.store.upload_file(
                                                model_path, model_script_key
                                            ):
                                                ctx.dataset_info["model_script_minio_key"] = model_script_key
                                                logger.info(
                                                    f"Model script available in MinIO: {model_script_key}"
                                                )
                                            else:
                                                logger.warning(
                                                    f"Failed to upload model script to MinIO: {model_path}"
                                                )
                                        else:
                                            logger.warning(
                                                f"Model script path not found for MinIO upload: {model_path}"
                                            )
                                except Exception as e:
                                    logger.warning(f"Failed to upload dataset to MinIO: {e}")
                        else:
                            raise RuntimeError(
                                f"Dataset preparation returned no dataset info for "
                                f"{cfg.dataset.name}/{cfg.dataset.variant}"
                            )
                except Exception as e:
                    logger.warning(f"Dataset context setup failed for run {run_id}: {e}")
                set_backend_context(ctx)

            # Step 4: Initialize scheduler
            state.initialize(pipeline, scheduler_type="priority")

            # Step 5: Mark RUNNING and sync to DB
            with session_scope() as session:
                run_repo = PipelineRunRepository(session)
                run_repo.update_status(run_id, PipelineRunStatus.RUNNING)

            if state.db_service and state.db_service.is_available():
                state.db_service.sync_pipeline_to_db(pipeline)

            logger.info(f"Run {run_id} initialized in background and marked RUNNING")
        except Exception as e:
            logger.error(f"Failed to initialize run {run_id} in background: {e}", exc_info=True)
            # Persist failure state so UI reflects startup errors.
            try:
                with session_scope() as session:
                    run_repo = PipelineRunRepository(session)
                    run_repo.update_status(run_id, PipelineRunStatus.FAILED, error_message=str(e))
            except Exception as db_e:
                logger.error(f"Failed to persist FAILED status for run {run_id}: {db_e}", exc_info=True)

    try:
        from src.backend.config_discovery import get_config_by_id
        from src.db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from src.pipeline.config_loader import (
            load_pipeline_config,
            load_attack_config,
            validate_pipeline_tool_dataset_compatibility,
        )
        from src.pipeline.stage_validation import load_tools_and_validate_pipeline_stages

        # Get config
        config = get_config_by_id(config_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Pipeline config '{config_id}' not found")

        effective_attack_config_path = request.attack_config_path or config.attack_config_path
        try:
            load_attack_config(effective_attack_config_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        config_path_obj = Path(config.config_path)
        if config_path_obj.exists():
            loaded_cfg = load_pipeline_config(config.config_path)
            effective_dataset_name = request.dataset_name or loaded_cfg.dataset.name
            if request.tools_override:
                for stage_name, tools in request.tools_override.items():
                    stage_cfg = loaded_cfg.pipeline.get(stage_name)
                    if stage_cfg is not None:
                        stage_cfg.tools = list(tools)

            tools_for_validation = load_tools_and_validate_pipeline_stages(
                loaded_cfg.pipeline,
                tools_yaml_path="configs/tools.yaml",
                fetch_remote_labels=True,
            )
            compatibility_issues = validate_pipeline_tool_dataset_compatibility(
                loaded_cfg,
                tools_for_validation,
                effective_dataset_name,
                fetch_remote_labels=True,
            )
            if compatibility_issues:
                first = compatibility_issues[0]
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Tool/dataset compatibility check failed: "
                        f"tool '{first['tool_id']}' ({first['tool_name']}) in stage '{first['stage']}' "
                        f"uses image '{first['image']}' which supports datasets [{first['supported_datasets']}], "
                        f"but requested dataset is '{first['requested_dataset']}'."
                    ),
                )
        else:
            logger.warning(
                "Skipping dataset compatibility check because config path does not exist: %s",
                config.config_path,
            )

        # ── Step 1: DB check + create run record (short-lived session) ──────────
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            active_runs = run_repo.get_active_runs_for_config(config_id)
            if active_runs:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot start new run: {len(active_runs)} active run(s) already exist for this config"
                )
            run_number = run_repo.get_next_run_number(config_id)
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            run_obj = run_repo.create({
                "id": run_id,
                "pipeline_config_id": config_id,
                "run_number": run_number,
                "use_cache": request.use_cache,
                "tools_config": request.tools_override,
                "status": PipelineRunStatus.PENDING,
            })
            # Capture fields before session closes
            run_id_val = run_obj.id
            run_number_val = run_obj.run_number
            run_use_cache = run_obj.use_cache
            run_tools_config = getattr(run_obj, "tools_config", None)
            run_status = run_obj.status.value
            run_error = run_obj.error_message
            run_created_at = run_obj.created_at.isoformat()
            run_started_at = run_obj.started_at.isoformat() if run_obj.started_at else None
        # ── session closed here; SQLite lock released ──────────────────────────

        # Heavy initialization moved to background task so UI/API call can return fast.
        request_data = request.model_dump()
        request_data["config_attack_config_path"] = config.attack_config_path
        asyncio.create_task(
            asyncio.to_thread(
                _start_pipeline_run_background,
                run_id_val,
                config.config_path,
                config.name,
                run_number_val,
                request_data,
                state,
            )
        )

        # Once initialization is successfully enqueued, report running to clients.
        # The DB row may still be pending for a brief moment until background
        # initialization updates it, but the API contract here is "accepted and active".
        response_status = (
            PipelineRunStatus.RUNNING.value
            if run_status == PipelineRunStatus.PENDING.value
            else run_status
        )

        return PipelineRunResponse(
            id=run_id_val,
            pipeline_config_id=config_id,
            run_number=run_number_val,
            use_cache=run_use_cache,
            tools_config=run_tools_config,
            status=response_status,
            error_message=run_error,
            created_at=run_created_at,
            started_at=run_started_at,
            completed_at=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start pipeline run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline run: {str(e)}")

@router.get("/api/pipeline-runs/{run_id}", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def get_pipeline_run(run_id: str):
    """Get status of a pipeline run."""
    try:
        from src.db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            return PipelineRunResponse(
                id=run.id,
                pipeline_config_id=run.pipeline_config_id,
                run_number=run.run_number,
                use_cache=run.use_cache,
                tools_config=getattr(run, "tools_config", None),
                status=run.status.value,
                error_message=run.error_message,
                created_at=run.created_at.isoformat(),
                started_at=run.started_at.isoformat() if run.started_at else None,
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline run: {str(e)}")

@router.get("/api/pipeline-runs/{run_id}/metrics", response_model=PipelineMetricsResponse, tags=["Metrics"])
async def get_pipeline_run_metrics(run_id: str):
    """
    Get evaluation metrics for all workflows in a historical pipeline run.

    Unlike /pipelines/{pipeline_id}/metrics this endpoint is purely DB-backed
    and does NOT require the scheduler to have this run's pipeline in memory.
    It works for any completed (or partially-completed) run.
    """
    try:
        from src.db import session_scope
        from src.db.models import EvaluationResultModel, PipelineRunModel, WorkflowModel

        with session_scope() as session:
            run = session.query(PipelineRunModel).filter(
                PipelineRunModel.id == run_id
            ).first()
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pipeline run '{run_id}' not found"
                )

            workflows = session.query(WorkflowModel).filter(
                WorkflowModel.run_id == run_id
            ).all()

            # Backward-compatibility fallback: older rows may only have pipeline_id set.
            if not workflows:
                workflows = session.query(WorkflowModel).filter(
                    WorkflowModel.pipeline_id == run_id
                ).all()

            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.run_id == run_id
            ).all()

            # Backward-compatibility fallback for legacy rows keyed only by pipeline_id.
            if not results:
                results = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == run_id
                ).all()

            # Return an empty-but-valid payload so UI can distinguish
            # "run exists, no metrics persisted yet" from true not-found.
            if not workflows and not results:
                return PipelineMetricsResponse(
                    pipeline_id=run_id,
                    pipeline_name=run_id,
                    workflow_count=0,
                    metric_names=[],
                    workflows=[],
                    summary={},
                )

            # Group evaluation results by workflow
            by_workflow: dict = {}
            metric_names: set = set()
            for r in results:
                if r.workflow_id not in by_workflow:
                    by_workflow[r.workflow_id] = {
                        "metrics": {},
                        "metric_sources": {},
                        "run": [],
                        "skipped": []
                    }
                if r.skipped:
                    by_workflow[r.workflow_id]["skipped"].append(r.evaluator_name)
                else:
                    by_workflow[r.workflow_id]["run"].append(r.evaluator_name)
                allowed_metrics = _allowed_metrics_for_evaluator(r.evaluator_name)
                for metric_name, value in (r.metrics or {}).items():
                    if allowed_metrics is not None and metric_name not in allowed_metrics:
                        continue
                    wf_bucket = by_workflow[r.workflow_id]
                    existing_source = wf_bucket["metric_sources"].get(metric_name)
                    if _should_override_metric(existing_source, r.evaluator_name, metric_name):
                        wf_bucket["metrics"][metric_name] = value
                        wf_bucket["metric_sources"][metric_name] = r.evaluator_name
                    metric_names.add(metric_name)

            # Build per-workflow metrics; determine is_baseline from DB task records.
            workflow_map = {wf.id: wf for wf in workflows}
            all_metrics = []
            for wf in workflows:
                non_eval_tasks = [t for t in wf.tasks if t.task_type != "evaluation"]
                is_baseline = bool(non_eval_tasks) and all(
                    t.tool_is_baseline for t in non_eval_tasks
                )
                wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
                workflow_tools = _workflow_tools_metadata(wf.tasks)
                all_metrics.append(WorkflowMetrics(
                    workflow_id=wf.id,
                    workflow_name=wf.name,
                    workflow_tools=workflow_tools["workflow_tools"],
                    workflow_tools_label=workflow_tools["workflow_tools_label"],
                    metrics=wf_data["metrics"],
                    evaluators_run=wf_data["run"],
                    evaluators_skipped=wf_data["skipped"],
                    is_baseline=is_baseline,
                ))

            # Include result-only workflows that may exist even when workflow rows are missing.
            for workflow_id, wf_data in by_workflow.items():
                if workflow_id in workflow_map:
                    continue
                all_metrics.append(WorkflowMetrics(
                    workflow_id=workflow_id,
                    workflow_name=workflow_id,
                    workflow_tools={},
                    workflow_tools_label="",
                    metrics=wf_data["metrics"],
                    evaluators_run=wf_data["run"],
                    evaluators_skipped=wf_data["skipped"],
                    is_baseline=False,
                ))

            # Compute summary statistics
            summary: dict = {}
            for metric_name in metric_names:
                values = [
                    wf.metrics.get(metric_name)
                    for wf in all_metrics
                    if wf.metrics.get(metric_name) is not None
                ]
                if values:
                    summary[metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values),
                    }
                else:
                    summary[metric_name] = {"min": None, "max": None, "avg": None, "count": 0}

            return PipelineMetricsResponse(
                pipeline_id=run_id,
                pipeline_name=run_id,
                workflow_count=len(workflows),
                metric_names=sorted(metric_names),
                workflows=all_metrics,
                summary=summary,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get run metrics: {str(e)}")

@router.get("/api/pipeline-configs/{config_id}/runs", response_model=PipelineRunListResponse, tags=["Pipeline Runs"])
async def get_pipeline_runs(config_id: str):
    """Get all runs for a pipeline config."""
    try:
        from src.db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            runs = run_repo.get_by_config_id(config_id)
            
            return PipelineRunListResponse(
                runs=[
                    PipelineRunResponse(
                        id=run.id,
                        pipeline_config_id=run.pipeline_config_id,
                        run_number=run.run_number,
                        use_cache=run.use_cache,
                        tools_config=getattr(run, "tools_config", None),
                        status=run.status.value,
                        error_message=run.error_message,
                        created_at=run.created_at.isoformat(),
                        started_at=run.started_at.isoformat() if run.started_at else None,
                        completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    )
                    for run in runs
                ],
                total=len(runs)
            )
    except Exception as e:
        logger.error(f"Failed to get pipeline runs for {config_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline runs: {str(e)}")

@router.get("/api/pipeline-runs", response_model=PipelineRunListResponse, tags=["Pipeline Runs"])
async def list_all_pipeline_runs(config_id: Optional[str] = Query(default=None)):
    """List all pipeline runs across all configs, optionally filtered by config_id."""
    try:
        from src.db import session_scope, PipelineRunRepository

        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            runs = run_repo.get_all(config_id=config_id)

            return PipelineRunListResponse(
                runs=[
                    PipelineRunResponse(
                        id=run.id,
                        pipeline_config_id=run.pipeline_config_id,
                        run_number=run.run_number,
                        use_cache=run.use_cache,
                        tools_config=getattr(run, "tools_config", None),
                        status=run.status.value,
                        error_message=run.error_message,
                        created_at=run.created_at.isoformat(),
                        started_at=run.started_at.isoformat() if run.started_at else None,
                        completed_at=run.completed_at.isoformat() if run.completed_at else None,
                    )
                    for run in runs
                ],
                total=len(runs)
            )
    except Exception as e:
        logger.error(f"Failed to list pipeline runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list pipeline runs: {str(e)}")

@router.post("/api/pipeline-runs/{run_id}/stop", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def stop_pipeline_run(
    run_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Stop a running pipeline."""
    try:
        from src.db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from src.pipeline.tasks import TaskStatus
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            if run.status not in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot stop run with status '{run.status.value}'"
                )
            
            # Update status to stopping
            run_repo.update_status(run_id, PipelineRunStatus.STOPPING)
            
            # Remove pending tasks from scheduler
            if scheduler and scheduler.pipeline and scheduler.pipeline.id == run_id:
                # Cancel all pending tasks for this run
                for task in scheduler._all_tasks:
                    if task.status == TaskStatus.PENDING and task.run_id == run_id:
                        task.status = TaskStatus.CANCELLED
                        if state.db_service:
                            state.db_service.sync_task_status(
                                task.id,
                                TaskStatus.CANCELLED,
                                error_message="Pipeline stopped by user"
                            )
            
            # Update status to cancelled
            run_repo.update_status(run_id, PipelineRunStatus.CANCELLED)
            
            # TODO: Trigger cache cleanup (will implement in cache cleanup task)
            
            run = run_repo.get_by_id(run_id)
            return PipelineRunResponse(
                id=run.id,
                pipeline_config_id=run.pipeline_config_id,
                run_number=run.run_number,
                use_cache=run.use_cache,
                tools_config=getattr(run, "tools_config", None),
                status=run.status.value,
                error_message=run.error_message,
                created_at=run.created_at.isoformat(),
                started_at=run.started_at.isoformat() if run.started_at else None,
                completed_at=run.completed_at.isoformat() if run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stop pipeline run: {str(e)}")

@router.post("/api/pipeline-runs/{run_id}/restart", response_model=PipelineRunResponse, tags=["Pipeline Runs"])
async def restart_pipeline_run(
    run_id: str,
    request: RestartPipelineRunRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Restart a stopped/failed pipeline run."""
    try:
        from src.db import get_session, session_scope, PipelineRunRepository, PipelineRunStatus
        from src.backend.config_discovery import get_config_by_id
        from src.pipeline.config_loader import create_pipeline_from_config, load_attack_config
        import uuid
        from datetime import datetime
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            old_run = run_repo.get_by_id(run_id)
            
            if not old_run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            # Check if old run is still active
            if old_run.status in (PipelineRunStatus.PENDING, PipelineRunStatus.RUNNING, PipelineRunStatus.STOPPING):
                raise HTTPException(
                    status_code=400,
                    detail="Cannot restart an active run. Stop it first."
                )
            
            # Get config
            config = get_config_by_id(old_run.pipeline_config_id)
            if not config:
                raise HTTPException(status_code=404, detail=f"Pipeline config '{old_run.pipeline_config_id}' not found")

            effective_attack_config_path = request.attack_config_path or config.attack_config_path
            try:
                load_attack_config(effective_attack_config_path)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # If not using cache, delete cache from previous run
            if not request.use_cache:
                # TODO: Implement cache deletion (will do in cache cleanup task)
                pass
            
            # Create new run
            run_number = run_repo.get_next_run_number(config.id)
            new_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            new_run = run_repo.create({
                "id": new_run_id,
                "pipeline_config_id": config.id,
                "run_number": run_number,
                "use_cache": request.use_cache,
                "status": PipelineRunStatus.PENDING,
            })
            
            # Load pipeline from config
            pipeline = create_pipeline_from_config(
                config_path=config.config_path,
                tools_yaml_path="configs/tools.yaml",
                evaluators_yaml_path="configs/evaluators.yaml",
                pipeline_name=f"{config.name} (Run {run_number})",
                clear_registry=True,
                include_evaluation=True,
                attack_config_path=effective_attack_config_path,
            )
            
            # Set pipeline ID to new run_id
            pipeline.id = new_run_id
            
            # Set run_id on all tasks and workflows
            for workflow in pipeline.workflows:
                workflow.pipeline_id = new_run_id
                workflow.run_id = new_run_id
                for task in workflow.tasks:
                    task.pipeline_id = new_run_id
                    task.run_id = new_run_id
            
            # Initialize scheduler with this pipeline
            state.initialize(pipeline, scheduler_type="priority")
            
            # Update run status to running
            run_repo.update_status(new_run_id, PipelineRunStatus.RUNNING)
            
            # Sync pipeline to DB
            if state.db_service and state.db_service.is_available():
                state.db_service.sync_pipeline_to_db(pipeline)
            
            return PipelineRunResponse(
                id=new_run.id,
                pipeline_config_id=new_run.pipeline_config_id,
                run_number=new_run.run_number,
                use_cache=new_run.use_cache,
                tools_config=getattr(new_run, "tools_config", None),
                status=new_run.status.value,
                error_message=new_run.error_message,
                created_at=new_run.created_at.isoformat(),
                started_at=new_run.started_at.isoformat() if new_run.started_at else None,
                completed_at=new_run.completed_at.isoformat() if new_run.completed_at else None,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart pipeline run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restart pipeline run: {str(e)}")

@router.delete("/api/pipeline-runs/{run_id}/cache", tags=["Pipeline Runs"])
async def delete_pipeline_run_cache(run_id: str):
    """Delete cache entries created by a specific run."""
    try:
        from src.db import get_session, session_scope, PipelineRunRepository, ArtifactRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            artifact_repo = ArtifactRepository(session)
            deleted_count = artifact_repo.delete_by_run_id(run_id)
            
            # TODO: Also delete from MinIO/local cache filesystem
            
            return {
                "success": True,
                "run_id": run_id,
                "deleted_artifacts": deleted_count,
                "message": f"Deleted {deleted_count} cache entries for run {run_id}"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete cache for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete cache: {str(e)}")

@router.delete("/api/pipeline-runs/{run_id}/workspace", tags=["Pipeline Runs"])
async def delete_pipeline_run_workspace(run_id: str):
    """Delete workspace files for a specific run."""
    try:
        from pathlib import Path
        import shutil
        
        from src.db import get_session, session_scope, PipelineRunRepository
        
        with session_scope() as session:
            run_repo = PipelineRunRepository(session)
            run = run_repo.get_by_id(run_id)
            
            if not run:
                raise HTTPException(status_code=404, detail=f"Pipeline run '{run_id}' not found")
            
            # Delete results directory
            results_dir = Path(f"results/{run.pipeline_config_id}/{run_id}")
            if results_dir.exists():
                shutil.rmtree(results_dir)
                return {
                    "success": True,
                    "run_id": run_id,
                    "deleted_path": str(results_dir),
                    "message": f"Deleted workspace for run {run_id}"
                }
            else:
                return {
                    "success": True,
                    "run_id": run_id,
                    "message": f"No workspace found for run {run_id}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workspace for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {str(e)}")
