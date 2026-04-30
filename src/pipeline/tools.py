"""
Tool definitions for Landseer pipeline.

This module defines the structure for tools used in the pipeline execution.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ContainerConfig(BaseModel):
    """Container configuration for a tool."""
    image: str = Field(description="Container image name")
    command: str = Field(description="Command to run the tool")
    runtime: Optional[str] = Field(default=None, description="Container runtime (docker/apptainer)")


class ToolDefinition(BaseModel):
    """Tool definition loaded from tools.yaml."""

    name: str = Field(description="Tool name")
    container: ContainerConfig = Field(description="Container configuration")
    is_baseline: bool = Field(default=False, description="Whether this tool is a baseline/noop tool")
    defense_stage: Optional[str] = Field(
        default=None,
        description=(
            "Pipeline stage this tool belongs to (synonym of pre_training, during_training, "
            "post_training, deployment). Alias key in YAML: 'stage'. Used for config validation; "
            "not inferred from tool id."
        ),
    )


Tool = ToolDefinition


_TOOL_REGISTRY: Dict[str, ToolDefinition] = {}


def load_tools_from_yaml(yaml_path: str) -> Dict[str, ToolDefinition]:
    """
    Load tool definitions from a YAML file.
    
    Args:
        yaml_path: Path to the tools.yaml configuration file
        
    Returns:
        Dictionary mapping tool names to ToolDefinition objects
    """
    import yaml
    from pathlib import Path
    
    tools = {}
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"Tools configuration file not found: {yaml_path}")
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data or 'tools' not in data:
        raise ValueError(f"Invalid tools.yaml format: missing 'tools' key")
    
    for tool_name, tool_data in data['tools'].items():
        tools[tool_name] = ToolDefinition(
            name=tool_data.get('name', tool_name),
            container=ContainerConfig(**tool_data['container']),
            is_baseline=tool_data.get('is_baseline', False),
            defense_stage=tool_data.get('defense_stage') or tool_data.get('stage'),
        )
    
    return tools


def init_tool_registry(yaml_path: str = "configs/tools.yaml"):
    """
    Initialize the global tool registry from a YAML file.
    
    Args:
        yaml_path: Path to the tools.yaml configuration file
    """
    global _TOOL_REGISTRY
    _TOOL_REGISTRY = load_tools_from_yaml(yaml_path)


def get_tool(tool_name: str) -> Optional[ToolDefinition]:
    """
    Get a tool definition from the registry.
    
    Args:
        tool_name: Name of the tool to retrieve
        
    Returns:
        ToolDefinition if found, None otherwise
    """
    return _TOOL_REGISTRY.get(tool_name)


def get_all_tools() -> Dict[str, ToolDefinition]:
    """
    Get all registered tools.
    
    Returns:
        Dictionary of all tool definitions
    """
    return _TOOL_REGISTRY.copy()
