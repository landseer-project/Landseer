"""Tools and evaluators registry endpoints."""
from __future__ import annotations
from src.backend.api.models import AddToolRequest, ContainerInfo, ToolInfo, ToolListResponse, EvaluatorInfo, EvaluatorListResponse, AddEvaluatorRequest

from fastapi import APIRouter, Depends

from src.common import get_logger
from src.backend.api.state import SchedulerState, get_scheduler_state
from src.backend.api.helpers import _require_pipeline_key, _tools_registry_path, _tool_key_from_name, _persist_tool_to_yaml

logger = get_logger(__name__)
router = APIRouter()

@router.get("/registry/tools", response_model=ToolListResponse, tags=["Registry"])
async def registry_list_tools(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all registered tools from the registry."""
    tools_dict = state.get_all_tools()

    tools = []
    for name, data in tools_dict.items():
        container = data.get("container", {})
        tools.append(ToolInfo(
            name=data.get("name", name),
            key=name,  # YAML registry key, used for tools_override
            container=ContainerInfo(
                image=container.get("image", ""),
                command=container.get("command", ""),
                runtime=container.get("runtime")
            ),
            is_baseline=data.get("is_baseline", False),
            defense_stage=data.get("defense_stage"),
        ))

    return ToolListResponse(tools=tools, total=len(tools))

@router.post("/registry/tools", response_model=ToolInfo, tags=["Registry"])
async def registry_add_tool(
    request: AddToolRequest,
    state: SchedulerState = Depends(get_scheduler_state),
    _auth: None = Depends(_require_pipeline_key),
):
    """
    Add a new tool to the registry.
    
    Persists to tools registry YAML and also updates runtime registry.
    """
    tool_key = _tool_key_from_name(request.name)
    _persist_tool_to_yaml(tool_key, request)
    try:
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(_tools_registry_path()))
    except Exception as exc:
        logger.warning("Failed to reload tool registry after persistence: %s", exc)

    state.add_tool(
        name=request.name,
        image=request.image,
        command=request.command,
        runtime=request.runtime,
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
        key=tool_key,
    )
    
    return ToolInfo(
        name=request.name,
        key=tool_key,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )

@router.get("/registry/evaluators", response_model=EvaluatorListResponse, tags=["Registry"])
async def registry_list_evaluators():
    """Get all registered evaluators from the registry."""
    from src.pipeline.config_loader import get_all_evaluators, init_evaluator_registry
    
    # Initialize if not already done
    try:
        init_evaluator_registry()
    except Exception:
        pass
    
    evaluators_dict = get_all_evaluators()
    
    evaluators = []
    for name, eval_def in evaluators_dict.items():
        evaluators.append(EvaluatorInfo(
            name=eval_def.name,
            container=ContainerInfo(
                image=eval_def.container.image,
                command=eval_def.container.command,
                runtime=eval_def.container.runtime
            ),
            required_artifacts=eval_def.required_artifacts,
            metrics=eval_def.metrics,
            defense_types=eval_def.defense_types
        ))
    
    return EvaluatorListResponse(evaluators=evaluators, total=len(evaluators))

@router.post("/registry/evaluators", response_model=EvaluatorInfo, tags=["Registry"])
async def registry_add_evaluator(request: AddEvaluatorRequest):
    """
    Add a new evaluator to the registry.
    
    Note: This is runtime only. For persistence, update configs/evaluators.yaml.
    """
    # For now, just return the evaluator info
    # Full persistence would require updating YAML file
    return EvaluatorInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        required_artifacts=request.required_artifacts,
        metrics=request.metrics,
        defense_types=request.defense_types
    )
