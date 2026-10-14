"""
Tool definitions for Landseer pipeline.

This module defines the structure for tools used in the pipeline execution.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Model framework of the pipeline's standard intermediate form (model.pt + config_model.py).
STANDARD_FRAMEWORK = "pytorch"


class ContainerConfig(BaseModel):
    """Container configuration for a tool."""
    image: str = Field(description="Container image name")
    command: str = Field(description="Command to run the tool")
    runtime: Optional[str] = Field(default=None, description="Container runtime (docker/apptainer)")


class AdapterDefinition(BaseModel):
    """Adapter container declared under the top-level ``adapters:`` key of tools.yaml.

    An adapter runs inside the same task as its tool: an input adapter before
    the tool (standard inputs -> the tool's format), an output adapter after it
    (the tool's outputs -> standard form, e.g. ``model.pt``). See
    docs/how-to/adapter-layers.md.
    """
    name: str = Field(description="Adapter name")
    source_framework: Optional[str] = Field(default=None, description="Framework the adapter reads")
    target_framework: Optional[str] = Field(default=None, description="Framework the adapter writes")
    container: ContainerConfig = Field(description="Container configuration")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment for the adapter container")


class ToolAdapters(BaseModel):
    """Optional adapters wrapped around a tool."""
    input: Optional[AdapterDefinition] = Field(default=None, description="Runs before the tool")
    output: Optional[AdapterDefinition] = Field(default=None, description="Runs after the tool")


class ToolDefinition(BaseModel):
    """Tool definition loaded from tools.yaml."""

    name: str = Field(description="Tool name")
    defense_stage: Optional[str] = Field(
        default=None,
        description=(
            "Pipeline stage this tool belongs to (synonym of pre_training, during_training, "
            "post_training, deployment). Alias key in YAML: 'stage'. Used for config validation; "
            "not inferred from tool id."
        ),)
    is_baseline: bool = Field(default=False, description="Whether this tool is a baseline/noop tool")
    container: ContainerConfig = Field(description="Container configuration")
    framework: Optional[str] = Field(
        default=None,
        description="Framework the tool works in (e.g. tensorflow). None means the standard (pytorch).",
    )
    adapters: Optional[ToolAdapters] = Field(
        default=None,
        description="Input/output adapters run in the same task as the tool",
    )


def adapter_task_config(tool: ToolDefinition) -> Dict[str, Any]:
    """
    Task config entries for a tool's adapters (empty when it has none).

    The adapters are carried in the task config so they are persisted with the
    task, reach the worker unchanged, and are part of the task hash, lineage
    and cache key like the rest of the config.
    """
    adapters = tool.adapters
    if adapters is None or (adapters.input is None and adapters.output is None):
        return {}
    spec: Dict[str, Any] = {}
    for position in ("input", "output"):
        adapter = getattr(adapters, position)
        if adapter is None:
            continue
        spec[position] = {
            "name": adapter.name,
            "image": adapter.container.image,
            "command": adapter.container.command,
            "env": dict(adapter.env),
        }
    return {"adapters": spec}


# Global tool registry - loaded at module import time
_TOOL_REGISTRY: Dict[str, ToolDefinition] = {}


def load_tools_from_yaml(yaml_path: str) -> Dict[str, ToolDefinition]:
    """
    Load tool definitions from a YAML file.
    
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

    adapter_defs = load_adapters(data)

    for tool_name, tool_data in data['tools'].items():
        tools[tool_name] = ToolDefinition(
            name=tool_data.get('name', tool_name),
            defense_stage=tool_data.get('defense_stage') or tool_data.get('stage'),
            is_baseline=tool_data.get('is_baseline', False),
            container=ContainerConfig(**tool_data['container']),
            framework=tool_data.get('framework'),
            adapters=_resolve_tool_adapters(tool_name, tool_data.get('adapters'), adapter_defs),
        )
        tool = tools[tool_name]
        if (
            tool.framework
            and tool.framework.lower() != STANDARD_FRAMEWORK
            and (tool.adapters is None or tool.adapters.output is None)
        ):
            logger.warning(
                f"Tool '{tool_name}' declares framework '{tool.framework}' but no output adapter; "
                f"downstream tasks expect {STANDARD_FRAMEWORK} outputs"
            )
    
    return tools


def load_adapters(data: Dict[str, Any]) -> Dict[str, AdapterDefinition]:
    """Parse the optional top-level ``adapters:`` section of a loaded tools.yaml."""
    raw = data.get('adapters') or {}
    if not isinstance(raw, dict):
        raise ValueError("Invalid tools.yaml format: 'adapters' must be a mapping")
    adapters = {}
    for adapter_name, adapter_data in raw.items():
        adapters[adapter_name] = AdapterDefinition(
            name=adapter_data.get('name', adapter_name),
            source_framework=adapter_data.get('source_framework'),
            target_framework=adapter_data.get('target_framework'),
            container=ContainerConfig(**adapter_data['container']),
            env={str(k): str(v) for k, v in (adapter_data.get('env') or {}).items()},
        )
    return adapters


def _resolve_tool_adapters(
    tool_name: str,
    raw: Any,
    adapter_defs: Dict[str, AdapterDefinition],
) -> Optional[ToolAdapters]:
    """Resolve a tool's ``adapters: {input: <name>, output: <name>}`` against the adapter section."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Tool '{tool_name}': 'adapters' must be a mapping with 'input'/'output' keys")
    unknown_keys = set(raw) - {'input', 'output'}
    if unknown_keys:
        raise ValueError(f"Tool '{tool_name}': unknown adapter position(s) {sorted(unknown_keys)}")
    resolved = {}
    for position in ('input', 'output'):
        adapter_name = raw.get(position)
        if not adapter_name:
            continue
        if adapter_name not in adapter_defs:
            raise ValueError(f"Tool '{tool_name}': {position} adapter '{adapter_name}' is not defined under 'adapters'")
        resolved[position] = adapter_defs[adapter_name]
    return ToolAdapters(**resolved) if resolved else None


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


# Backward-compatible alias used by package exports / older callers.
Tool = ToolDefinition
