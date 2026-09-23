"""Tool registry endpoints."""
from __future__ import annotations
from src.backend.api.models import AddToolRequest, ContainerInfo, ToolInfo, ToolListResponse

from fastapi import APIRouter, Depends, HTTPException

from src.common import get_logger
from src.backend.api.state import SchedulerState, get_scheduler_state

logger = get_logger(__name__)
router = APIRouter()

@router.get("/tools", response_model=ToolListResponse, tags=["Tools"])
async def list_tools(state: SchedulerState = Depends(get_scheduler_state)):
    """Get all available tools."""
    tools_dict = state.get_all_tools()

    tools = []
    for name, data in tools_dict.items():
        container = data.get("container", {})
        tools.append(ToolInfo(
            name=data.get("name", name),
            container=ContainerInfo(
                image=container.get("image", ""),
                command=container.get("command", ""),
                runtime=container.get("runtime")
            ),
            is_baseline=data.get("is_baseline", False),
            defense_stage=data.get("defense_stage"),
        ))

    return ToolListResponse(tools=tools, total=len(tools))

@router.get("/tools/{tool_name}", response_model=ToolInfo, tags=["Tools"])
async def get_tool(
    tool_name: str,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """Get information about a specific tool."""
    tools = state.get_all_tools()
    
    if tool_name not in tools:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    data = tools[tool_name]
    container = data.get("container", {})
    
    return ToolInfo(
        name=data.get("name", tool_name),
        container=ContainerInfo(
            image=container.get("image", ""),
            command=container.get("command", ""),
            runtime=container.get("runtime")
        ),
        is_baseline=data.get("is_baseline", False)
    )

@router.post("/tools", response_model=ToolInfo, tags=["Tools"])
async def add_tool(
    request: AddToolRequest,
    state: SchedulerState = Depends(get_scheduler_state)
):
    """
    Add a new tool to the pipeline.
    
    Note: This adds the tool to the runtime registry only.
    For persistent tools, update configs/tools.yaml.
    """
    state.add_tool(
        name=request.name,
        image=request.image,
        command=request.command,
        runtime=request.runtime,
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )
    
    return ToolInfo(
        name=request.name,
        container=ContainerInfo(
            image=request.image,
            command=request.command,
            runtime=request.runtime
        ),
        is_baseline=request.is_baseline,
        defense_stage=request.defense_stage,
    )
