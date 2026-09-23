"""Shared helpers for API routes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import yaml
from fastapi import Header, HTTPException

from src.common import get_logger
from .models import AddToolRequest, ContainerInfo, TaskResponse, ToolInfo
from .state import SchedulerState

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

def task_to_response(task, state: Optional["SchedulerState"] = None) -> TaskResponse:
    """Convert a Task object to a TaskResponse model."""
    # Get workflow names from pipeline
    workflow_names = []
    if state and state.pipeline:
        for workflow in state.pipeline.workflows:
            if workflow.id in task.workflows:
                workflow_names.append(workflow.name)
    
    # Get execution metadata from task metadata
    cache_hit = None
    cache_key = None
    output_path = None
    log_path = None
    worker_id = None
    error_message = None
    execution_time_ms = None
    
    if state:
        metadata = state.task_metadata.get(task.id, {})
        worker_id = metadata.get("worker_id")
        error_message = metadata.get("error_message")
        execution_time_ms = metadata.get("execution_time_ms")
        result = metadata.get("result", {})
        if isinstance(result, dict):
            cache_hit = result.get("cache_hit")
            cache_key = result.get("cache_key")
            output_path = result.get("output_path")
            log_path = result.get("log_path")
    
    # Get run_id from task
    run_id = getattr(task, 'run_id', None)
    
    return TaskResponse(
        id=task.id,
        tool=ToolInfo(
            name=task.tool.name,
            container=ContainerInfo(
                image=task.tool.container.image,
                command=task.tool.container.command,
                runtime=task.tool.container.runtime
            ),
            is_baseline=task.tool.is_baseline
        ),
        config=task.config,
        priority=task.priority,
        status=task.status.value,
        task_type=task.task_type.value,
        counter=task.counter,
        workflows=list(task.workflows),
        workflow_names=workflow_names,
        pipeline_id=task.pipeline_id,
        run_id=run_id,
        dependency_ids=[dep.id for dep in task.dependencies],
        cache_hit=cache_hit,
        cache_key=cache_key,
        output_path=output_path,
        log_path=log_path,
        worker_id=worker_id,
        error_message=error_message,
        execution_time_ms=execution_time_ms
    )


def _required_pipeline_keys() -> List[str]:
    """Read configured API keys used to protect mutating endpoints."""
    raw = os.getenv("LANDSEER_PIPELINE_KEYS", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _require_pipeline_key(x_pipeline_key: Optional[str] = Header(default=None, alias="X-Pipeline-Key")) -> None:
    """
    Enforce the same key gate as run-start when LANDSEER_PIPELINE_KEYS is configured.
    If no keys are configured, endpoint remains open for local/dev workflows.
    """
    allowed = _required_pipeline_keys()
    if not allowed:
        return
    if not x_pipeline_key or x_pipeline_key not in allowed:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Pipeline-Key")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tools_registry_path() -> Path:
    """
    Resolve persistent tools registry YAML path.
    Allows override via LANDSEER_TOOLS_YAML for tests or custom deployments.
    """
    cfg = os.getenv("LANDSEER_TOOLS_YAML", "configs/tools.yaml")
    p = Path(cfg)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return _repo_root() / p


def _tool_key_from_name(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "custom_tool"


def _persist_tool_to_yaml(tool_key: str, request: AddToolRequest) -> None:
    """Persist a tool entry to tools.yaml so registry survives backend restart."""
    yaml_path = _tools_registry_path()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    tools = data.get("tools")
    if not isinstance(tools, dict):
        tools = {}

    tools[tool_key] = {
        "name": request.name,
        "defense_stage": request.defense_stage,
        "is_baseline": request.is_baseline,
        "container": {
            "image": request.image,
            "command": request.command,
            "runtime": request.runtime,
        },
    }
    data["tools"] = tools

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
