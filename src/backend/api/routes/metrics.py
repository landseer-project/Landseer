"""Pipeline and workflow metrics endpoints."""
from __future__ import annotations
from src.backend.api.models import PipelineMetricsResponse, WorkflowMetrics

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.common import get_logger
from src.pipeline.tasks import TaskType
from src.backend.scheduler import Scheduler
from src.backend.api.state import SchedulerState, get_scheduler, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()

def _task_type_token(task_type: Any) -> str:
    """Normalize TaskType enums/strings to a lowercase stage token."""
    if task_type is None:
        return ""
    if isinstance(task_type, TaskType):
        return task_type.value
    value = getattr(task_type, "value", None)
    if isinstance(value, str):
        return value
    text = str(task_type).strip()
    prefix = "tasktype."
    lowered = text.lower()
    if lowered.startswith(prefix):
        return lowered[len(prefix):]
    return lowered

def _stage_rank(task_type: Any) -> int:
    """Sort task stages in pipeline order, putting unknowns at the end."""
    key = _task_type_token(task_type)
    order = {
        TaskType.PRE_TRAINING.value: 0,
        "pre": 0,
        TaskType.IN_TRAINING.value: 1,
        "in": 1,
        "during": 1,
        "during_training": 1,
        TaskType.POST_TRAINING.value: 2,
        "post": 2,
        "post_training": 2,
        TaskType.DEPLOYMENT.value: 3,
        "deploy": 3,
        "deployment": 3,
    }
    return order.get(key, 99)

def _stage_label(task_type: Any) -> str:
    """Map task type to a user-friendly short stage label."""
    key = _task_type_token(task_type)
    mapping = {
        TaskType.PRE_TRAINING.value: "pre",
        "pre": "pre",
        TaskType.IN_TRAINING.value: "in",
        "in": "in",
        "during": "in",
        "during_training": "in",
        TaskType.POST_TRAINING.value: "post",
        "post": "post",
        "post_training": "post",
        TaskType.DEPLOYMENT.value: "deploy",
        "deploy": "deploy",
        "deployment": "deploy",
    }
    return mapping.get(key, key)

def _task_tool_name(task: Any) -> Optional[str]:
    tool_name = getattr(task, "tool_name", None)
    if tool_name:
        return str(tool_name)
    tool_obj = getattr(task, "tool", None)
    name = getattr(tool_obj, "name", None)
    if name:
        return str(name)
    config = getattr(task, "config", None) or {}
    config_name = config.get("tool_name")
    return str(config_name) if config_name else None

def _task_stage_label(task: Any) -> str:
    task_type = getattr(task, "task_type", "")
    stage = _stage_label(task_type)
    if stage in {"pre", "in", "post", "deploy"}:
        return stage
    config = getattr(task, "config", None) or {}
    return _stage_label(config.get("stage", ""))

def _join_stage_tools(names: List[str]) -> str:
    """Render a stage's tools in execution order for CSV/display."""
    return " -> ".join(names)

def _workflow_tools_metadata(tasks: List[Any]) -> Dict[str, Any]:
    """Return grouped stage tools and a compact label for workflow display.

    Tool names are kept in execution order: stages pre → in → post → deploy,
    and within a stage the sequence the combo actually runs.
    """
    grouped: Dict[str, List[str]] = {stage: [] for stage in ("pre", "in", "post", "deploy")}
    ordered = sorted(
        enumerate(tasks),
        key=lambda item: (_stage_rank(getattr(item[1], "task_type", "")), item[0]),
    )
    for _, task in ordered:
        if _task_type_token(getattr(task, "task_type", "")) == TaskType.EVALUATION.value:
            continue
        stage = _task_stage_label(task)
        if stage not in grouped:
            continue
        tool_name = _task_tool_name(task)
        if not tool_name:
            continue
        grouped[stage].append(tool_name)

    label_parts: List[str] = []
    for stage in ("pre", "in", "post", "deploy"):
        tools = grouped.get(stage, [])
        if not tools:
            continue
        label_parts.append(f"{stage}: {_join_stage_tools(tools)}")

    return {
        "workflow_tools": grouped,
        "workflow_tools_label": " | ".join(label_parts),
    }

def _allowed_metrics_for_evaluator(evaluator_name: str) -> Optional[set]:
    """
    Return the allowlist of metric names for an evaluator.

    This prevents one evaluator from overwriting another evaluator's metrics
    with undeclared keys (e.g., fingerprinting emitting clean_accuracy).
    """
    try:
        from src.pipeline.config_loader import get_all_evaluators
        evaluators = get_all_evaluators()
        evaluator = evaluators.get(evaluator_name)
        if evaluator and evaluator.metrics:
            return set(evaluator.metrics)
    except Exception:
        pass
    return None

def _should_override_metric(existing_evaluator: Optional[str], new_evaluator: str, metric_name: str) -> bool:
    """
    Decide overwrite behavior when multiple evaluators emit the same metric key.

    clean_accuracy may appear in both clean and adversarial evaluators; prefer
    the clean evaluator value for dashboard consistency.
    """
    if existing_evaluator is None:
        return True
    if metric_name == "clean_accuracy":
        if existing_evaluator == "clean":
            return False
        if new_evaluator == "clean":
            return True
    return False

def _export_run_metrics_csv(run_id: str, scheduler: Scheduler) -> Optional[Path]:
    """
    Export per-workflow evaluation metrics to CSV for a completed run.

    Every expected evaluator is included. If evaluator output is missing, skipped,
    or failed, metric values are written as -1.
    """
    try:
        from src.db import session_scope
        from src.db.models import EvaluationResultModel, PipelineRunModel
    except Exception as e:
        logger.warning(f"Unable to import DB models for metrics CSV export: {e}")
        return None

    pipeline = getattr(scheduler, "pipeline", None)
    if not pipeline:
        return None

    workflow_name_by_id: Dict[str, str] = {wf.id: wf.name for wf in pipeline.workflows}
    workflow_is_baseline: Dict[str, bool] = {}
    expected_evaluators_by_workflow: Dict[str, set] = {}
    expected_metric_names_by_evaluator: Dict[str, set] = {}

    for wf in pipeline.workflows:
        non_eval_tasks = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
        workflow_is_baseline[wf.id] = bool(non_eval_tasks) and all(
            getattr(t.tool, "is_baseline", False) for t in non_eval_tasks
        )
        for task in wf.tasks:
            if task.task_type != TaskType.EVALUATION:
                continue
            evaluator_name = str(task.tool.name)
            expected_evaluators_by_workflow.setdefault(wf.id, set()).add(evaluator_name)
            expected_metric_names_by_evaluator.setdefault(evaluator_name, set()).update(
                [str(m) for m in (task.config or {}).get("metrics", []) if str(m).strip()]
            )

    if not expected_evaluators_by_workflow:
        logger.info(f"Run {run_id}: no evaluation tasks found; skipping metrics CSV export")
        return None

    with session_scope() as session:
        run = session.query(PipelineRunModel).filter(PipelineRunModel.id == run_id).first()

        results = session.query(EvaluationResultModel).filter(
            EvaluationResultModel.run_id == run_id
        ).all()

        # Backward compatibility for rows keyed only by pipeline_id.
        if not results:
            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.pipeline_id == run_id
            ).all()
 
    if run is not None:
        config_id = run.pipeline_config_id
    else:
        pipeline_name = getattr(pipeline, "name", None) or "unknown_config"
        config_id = (
            pipeline_name
            if str(pipeline_name).startswith("config_")
            else f"config_{pipeline_name}"
        )
        logger.warning(
            f"Run {run_id} not found in pipeline_runs; exporting CSV under {config_id}/{run_id}"
        )

    result_lookup: Dict[tuple, Any] = {}
    observed_metric_names_by_evaluator:  Dict[str, set] = {}
    for result in results:
        key = (result.workflow_id, result.evaluator_name)
        result_lookup[key] = result
        observed_metric_names_by_evaluator.setdefault(result.evaluator_name, set()).update(
            list((result.metrics or {}).keys())
        )

    metric_names_by_evaluator: Dict[str, List[str]] = {}
    for evaluator_name in set(expected_metric_names_by_evaluator) | set(observed_metric_names_by_evaluator):
        merged = (
            expected_metric_names_by_evaluator.get(evaluator_name, set())
            | observed_metric_names_by_evaluator.get(evaluator_name, set())
        )
        metric_names_by_evaluator[evaluator_name] = sorted(merged)

    all_evaluators = sorted(
        set(expected_metric_names_by_evaluator.keys())
        | set(observed_metric_names_by_evaluator.keys())
    )
    if not all_evaluators:
        return None

    output_dir = Path("results") / config_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_summary.csv"

    workflow_by_id = {wf.id: wf for wf in pipeline.workflows}
    header: List[str] = [
        "run_id",
        "workflow_id",
        "workflow_name",
        "pre",
        "in",
        "post",
        "deploy",
        "pre_training",
        "in_training",
        "post_training",
        "deployment",
        "is_baseline",
        "combination_success",
    ]
    for evaluator_name in all_evaluators:
        header.append(f"{evaluator_name}.status")
        for metric_name in metric_names_by_evaluator.get(evaluator_name, []):
            header.append(f"{evaluator_name}.{metric_name}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for workflow_id in sorted(expected_evaluators_by_workflow.keys()):
            wf = workflow_by_id.get(workflow_id)
            tools_meta = _workflow_tools_metadata(wf.tasks if wf else [])
            tools = tools_meta["workflow_tools"]
            row: Dict[str, Any] = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow_name_by_id.get(workflow_id, workflow_id),
                "pre": _join_stage_tools(tools.get("pre", [])),
                "in": _join_stage_tools(tools.get("in", [])),
                "post": _join_stage_tools(tools.get("post", [])),
                "deploy": _join_stage_tools(tools.get("deploy", [])),
                "pre_training": _join_stage_tools(tools.get("pre", [])),
                "in_training": _join_stage_tools(tools.get("in", [])),
                "post_training": _join_stage_tools(tools.get("post", [])),
                "deployment": _join_stage_tools(tools.get("deploy", [])),
                "is_baseline": workflow_is_baseline.get(workflow_id, False),
            }
            expected_evaluators = expected_evaluators_by_workflow.get(workflow_id, set())
            statuses: List[str] = []
            for evaluator_name in all_evaluators:
                result = result_lookup.get((workflow_id, evaluator_name))
                if evaluator_name not in expected_evaluators:
                    status = "not_applicable"
                elif result is None:
                    status = "missing"
                elif result.skipped:
                    status = "skipped"
                elif not result.success:
                    status = "failed"
                else:
                    status = "ok"
                statuses.append(status)
                row[f"{evaluator_name}.status"] = status

                for metric_name in metric_names_by_evaluator.get(evaluator_name, []):
                    col = f"{evaluator_name}.{metric_name}"
                    if status != "ok":
                        row[col] = -1
                        continue
                    metric_val = (result.metrics or {}).get(metric_name) if result else None
                    row[col] = metric_val if metric_val is not None else -1

            # success if nothing expected failed/missing (skipped is ok)
            relevant = [s for s, name in zip(statuses, all_evaluators) if name in expected_evaluators]
            row["combination_success"] = (
                "success"
                if relevant and all(s in ("ok", "skipped") for s in relevant)
                else "failure"
            )
            writer.writerow(row)

    return csv_path

@router.get("/pipelines/{pipeline_id}/metrics", response_model=PipelineMetricsResponse, tags=["Metrics"])
async def get_pipeline_metrics(
    pipeline_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """
    Get evaluation metrics for all workflows in a pipeline.
    
    Returns metrics from all evaluators across all workflows,
    including summary statistics.
    """
    pipeline = scheduler.pipeline
    
    if pipeline.id != pipeline_id and pipeline.name != pipeline_id:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    
    # Try to get metrics from database
    all_metrics = []
    metric_names = set()
    db_matched = False  # True only when DB has rows matching current workflow IDs

    if state.db_service and state.db_service.is_available():
        # Get from database
        try:
            from src.db.models import EvaluationResultModel
            from src.db import session_scope

            current_workflow_ids = {wf.id for wf in pipeline.workflows}

            with session_scope() as session:
                results = session.query(EvaluationResultModel).filter(
                    EvaluationResultModel.pipeline_id == pipeline.id,
                    EvaluationResultModel.workflow_id.in_(current_workflow_ids)
                ).all()

                logger.debug(f"Found {len(results)} evaluation results in database for pipeline {pipeline.id}")

                if results:
                    db_matched = True
                    # Group by workflow
                    by_workflow = {}
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

                    for wf in pipeline.workflows:
                        wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
                        non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
                        workflow_tools = _workflow_tools_metadata(wf.tasks)
                        all_metrics.append(WorkflowMetrics(
                            workflow_id=wf.id,
                            workflow_name=wf.name,
                            workflow_tools=workflow_tools["workflow_tools"],
                            workflow_tools_label=workflow_tools["workflow_tools_label"],
                            metrics=wf_data["metrics"],
                            evaluators_run=wf_data["run"],
                            evaluators_skipped=wf_data["skipped"],
                            is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
                        ))

        except Exception as e:
            logger.warning(f"Failed to get metrics from database: {e}", exc_info=True)

    # If database had no rows matching current workflow IDs, fall back to in-memory task metadata.
    # This handles: DB unavailable, fresh backend restart (new sequential IDs), or stale DB rows
    # from a previous run that used different workflow IDs.
    if not db_matched:
        by_workflow = {
            wf.id: {"metrics": {}, "run": [], "skipped": []} for wf in pipeline.workflows
        }
        
        # Walk over all evaluation tasks and extract evaluation_result from task metadata
        for wf in pipeline.workflows:
            for task in wf.tasks:
                if task.task_type != TaskType.EVALUATION:
                    continue
                
                meta = state.task_metadata.get(task.id) or {}
                result = meta.get("result") or {}
                eval_result = result.get("evaluation_result")
                if not isinstance(eval_result, dict):
                    continue
                
                evaluator_name = task.tool.name
                metrics_dict = eval_result.get("metrics") or {}
                skipped = bool(eval_result.get("skipped"))
                
                wf_data = by_workflow[wf.id]
                if skipped:
                    wf_data["skipped"].append(evaluator_name)
                else:
                    wf_data["run"].append(evaluator_name)
                for metric_name, value in metrics_dict.items():
                    try:
                        numeric_val = float(value)
                    except (TypeError, ValueError):
                        continue
                    wf_data["metrics"][metric_name] = numeric_val
                    metric_names.add(metric_name)
        
        for wf in pipeline.workflows:
            wf_data = by_workflow.get(wf.id, {"metrics": {}, "run": [], "skipped": []})
            non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
            workflow_tools = _workflow_tools_metadata(wf.tasks)
            all_metrics.append(WorkflowMetrics(
                workflow_id=wf.id,
                workflow_name=wf.name,
                workflow_tools=workflow_tools["workflow_tools"],
                workflow_tools_label=workflow_tools["workflow_tools_label"],
                metrics=wf_data["metrics"],
                evaluators_run=wf_data["run"],
                evaluators_skipped=wf_data["skipped"],
                is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
            ))

    # If still no results, return empty metrics
    if not all_metrics:
        for wf in pipeline.workflows:
            non_eval = [t for t in wf.tasks if t.task_type != TaskType.EVALUATION]
            workflow_tools = _workflow_tools_metadata(wf.tasks)
            all_metrics.append(WorkflowMetrics(
                workflow_id=wf.id,
                workflow_name=wf.name,
                workflow_tools=workflow_tools["workflow_tools"],
                workflow_tools_label=workflow_tools["workflow_tools_label"],
                metrics={},
                evaluators_run=[],
                evaluators_skipped=[],
                is_baseline=bool(non_eval) and all(t.tool.is_baseline for t in non_eval),
            ))
    
    # Calculate summary statistics
    summary = {}
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
                "count": len(values)
            }
        else:
            summary[metric_name] = {"min": None, "max": None, "avg": None, "count": 0}
    
    return PipelineMetricsResponse(
        pipeline_id=pipeline.id,
        pipeline_name=pipeline.name,
        workflow_count=len(pipeline.workflows),
        metric_names=sorted(metric_names),
        workflows=all_metrics,
        summary=summary
    )

@router.get("/workflows/{workflow_id}/metrics", tags=["Metrics"])
async def get_workflow_metrics(
    workflow_id: str,
    state: SchedulerState = Depends(get_scheduler_state),
    scheduler: Scheduler = Depends(get_scheduler)
):
    """Get evaluation metrics for a specific workflow."""
    # Find workflow
    workflow = None
    for w in scheduler.pipeline.workflows:
        if w.id == workflow_id or w.name == workflow_id:
            workflow = w
            break
    
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    metrics = {}
    evaluators_run = []
    evaluators_skipped = []
    
    # Try to get from database
    if state.db_service and state.db_service.is_available():
        try:
            from src.db.models import EvaluationResultModel
            session = state.db_service.get_session()
            results = session.query(EvaluationResultModel).filter(
                EvaluationResultModel.workflow_id == workflow.id
            ).all()
            
            for r in results:
                if r.skipped:
                    evaluators_skipped.append({
                        "evaluator": r.evaluator_name,
                        "reason": r.skip_reason
                    })
                else:
                    evaluators_run.append(r.evaluator_name)
                    metrics.update(r.metrics or {})
                    
        except Exception as e:
            logger.warning(f"Failed to get metrics from database: {e}")
    
    return {
        "workflow_id": workflow.id,
        "workflow_name": workflow.name,
        "pipeline_id": workflow.pipeline_id,
        "metrics": metrics,
        "evaluators_run": evaluators_run,
        "evaluators_skipped": evaluators_skipped
    }
