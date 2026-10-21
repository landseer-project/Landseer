"""
Tool definitions for Landseer pipeline.

This module defines the structure for tools used in the pipeline execution.
"""

import os
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Model framework of the pipeline's standard intermediate form (model.pt + config_model.py).
STANDARD_FRAMEWORK = "pytorch"

# What a tool reads and writes, from its org.landseer.model_input / model_output
# labels (or the same keys in tools.yaml). See docs/reference/image-labels.md.
MODEL_INPUT_VALUES = ("none", "architecture", "weights")
MODEL_OUTPUT_VALUES = ("none", "model")

# Where a tool in a given framework reads/writes its model when it declares no
# org.landseer.model_path (relative to /input and /output).
DEFAULT_MODEL_PATHS = {"tensorflow": "model_tf/model.keras"}

_FRAMEWORK_ALIASES = {"torch": "pytorch", "torchvision": "pytorch", "tf": "tensorflow", "keras": "tensorflow"}

# Set to 0 to never look up tool image labels for adapter selection.
AUTO_ADAPTERS_ENV = "LANDSEER_AUTO_ADAPTERS"


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
    auto_select: bool = Field(
        default=True,
        description="May be picked automatically for tools that declare no adapters (matched on frameworks)",
    )


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
        description="Input/output adapters run in the same task as the tool (overrides automatic selection)",
    )
    model_input: Optional[str] = Field(
        default=None,
        description="Overrides the image's org.landseer.model_input (none/architecture/weights)",
    )
    model_output: Optional[str] = Field(
        default=None,
        description="Overrides the image's org.landseer.model_output (none/model)",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Overrides the image's org.landseer.model_path (the tool's model file, relative to /input and /output)",
    )
    # Adapters of the tools.yaml this tool came from, for automatic selection.
    # Not part of the tool's identity: the selected adapters land in the task config.
    adapter_catalog: Dict[str, AdapterDefinition] = Field(default_factory=dict, exclude=True)


# ---------------------------------------------------------------------------
# Adapter selection
# ---------------------------------------------------------------------------

def normalize_framework(value: Optional[str]) -> Optional[str]:
    """``tensorflow==2.15.0`` -> ``tensorflow``; for a list (``pytorch==2.1,tensorflow``) the first entry."""
    if not value:
        return None
    first = str(value).split(",")[0].strip().lower()
    for sep in ("==", ">=", "<=", "~=", "=", " "):
        first = first.split(sep)[0]
    first = first.strip()
    return _FRAMEWORK_ALIASES.get(first, first) or None


_LABEL_CACHE: Dict[tuple, Dict[str, str]] = {}


def _fetch_image_labels(image: str, runtime: Optional[str] = None) -> Dict[str, str]:
    """Labels of a tool image (empty when they cannot be fetched).

    Found labels are cached per process; failed lookups are not, so a
    registry outage does not stick.
    """
    key = (image, runtime)
    if key not in _LABEL_CACHE:
        from .container_labels import get_container_labels_for_image
        labels = get_container_labels_for_image(image, runtime=runtime)
        if not labels:
            return {}
        _LABEL_CACHE[key] = dict(labels)
    return dict(_LABEL_CACHE[key])


def _needs_labels(tool: ToolDefinition) -> bool:
    """Whether resolving the tool's adapters needs anything its tools.yaml entry does not say."""
    if os.environ.get(AUTO_ADAPTERS_ENV, "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    if not tool.adapter_catalog:
        # Nothing could be selected (e.g. a tool built in code, or a tools.yaml without adapters).
        return False
    framework = normalize_framework(tool.framework)
    if framework == STANDARD_FRAMEWORK:
        return False
    if tool.adapters is not None:
        # Explicit adapters; labels only fill in the model path for the output check.
        return tool.adapters.output is not None and not tool.model_path and framework not in DEFAULT_MODEL_PATHS
    return not (framework and tool.model_input and tool.model_output and tool.model_path)


def _pick_adapter(
    catalog: Dict[str, AdapterDefinition], source: str, target: str, tool_name: str, position: str,
) -> AdapterDefinition:
    candidates = [
        (key, adapter) for key, adapter in sorted(catalog.items())
        if adapter.auto_select
        and source in _split_frameworks(adapter.source_framework)
        and target in _split_frameworks(adapter.target_framework)
    ]
    if not candidates:
        raise ValueError(
            f"Tool '{tool_name}' needs a {source} -> {target} {position} adapter, but tools.yaml defines none "
            f"(declare one under 'adapters:' with source_framework: {source} and target_framework: {target})"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Tool '{tool_name}': several {source} -> {target} adapters could be picked automatically "
            f"({', '.join(k for k, _ in candidates)}); set auto_select: false on all but one, "
            f"or name the adapter under the tool's 'adapters:'"
        )
    return candidates[0][1]


def _split_frameworks(value: Optional[str]) -> List[str]:
    return [normalize_framework(p) for p in str(value or "").split(",") if p.strip()]


def _adapted_frameworks(catalog: Dict[str, AdapterDefinition]) -> set:
    """Non-standard frameworks some adapter in the catalog converts from or to."""
    found = set()
    for adapter in catalog.values():
        found.update(_split_frameworks(adapter.source_framework))
        found.update(_split_frameworks(adapter.target_framework))
    found.discard(STANDARD_FRAMEWORK)
    return found


def resolve_adapters(
    tool: ToolDefinition,
    label_fetcher: Optional[Callable[[str, Optional[str]], Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    The adapters a tool's task runs, as stored in the task config (``{}`` for none).

    Explicit ``adapters:`` in tools.yaml are used as they are. Otherwise the
    tool's framework, model_input, model_output and model_path come from
    tools.yaml or, when it does not say, from the image's ``org.landseer.*``
    labels, and adapters are picked from the tools.yaml ``adapters:`` section
    by framework: an input adapter (pytorch -> framework) unless model_input is
    ``none``, an output adapter (framework -> pytorch) when model_output is
    ``model``.

    The output adapter's ``model_path`` lets the worker check that the tool
    really wrote its model before converting it back.

    Raises:
        ValueError: On invalid label values, a non-standard framework without
            model_input/model_output, or no (or several) matching adapters.
    """
    from .image_labels import NS

    labels: Dict[str, str] = {}
    if _needs_labels(tool):
        fetch = label_fetcher or _fetch_image_labels
        labels = fetch(tool.container.image, tool.container.runtime) or {}

    framework = normalize_framework(tool.framework or labels.get(NS + "framework"))
    model_input = (tool.model_input or labels.get(NS + "model_input") or "").strip().lower() or None
    model_output = (tool.model_output or labels.get(NS + "model_output") or "").strip().lower() or None
    model_path = (tool.model_path or labels.get(NS + "model_path") or "").strip() or None
    if model_input and model_input not in MODEL_INPUT_VALUES:
        raise ValueError(f"Tool '{tool.name}': model_input {model_input!r} not in {MODEL_INPUT_VALUES}")
    if model_output and model_output not in MODEL_OUTPUT_VALUES:
        raise ValueError(f"Tool '{tool.name}': model_output {model_output!r} not in {MODEL_OUTPUT_VALUES}")
    if model_path is None and framework:
        model_path = DEFAULT_MODEL_PATHS.get(framework)
    if model_path and (model_path.startswith("/") or ".." in model_path.split("/")):
        raise ValueError(f"Tool '{tool.name}': model_path {model_path!r} must be relative to /input and /output")

    if tool.adapters is not None and (tool.adapters.input or tool.adapters.output):
        selected = {"input": tool.adapters.input, "output": tool.adapters.output}
    elif not framework or framework == STANDARD_FRAMEWORK:
        return {}
    elif not model_input and not model_output and framework not in _adapted_frameworks(tool.adapter_catalog):
        # e.g. numpy / scikit-learn data tools: no model in another framework to convert.
        return {}
    else:
        if not model_input or not model_output:
            raise ValueError(
                f"Tool '{tool.name}' uses framework '{framework}', but neither its image labels "
                f"({NS}model_input/{NS}model_output; image {tool.container.image}"
                f"{'' if labels else ', labels could not be fetched'}) nor tools.yaml "
                f"(model_input/model_output) say which model it reads and writes, so its adapters cannot be chosen"
            )
        selected = {"input": None, "output": None}
        if model_input != "none":
            selected["input"] = _pick_adapter(tool.adapter_catalog, STANDARD_FRAMEWORK, framework, tool.name, "input")
        if model_output == "model":
            selected["output"] = _pick_adapter(tool.adapter_catalog, framework, STANDARD_FRAMEWORK, tool.name, "output")
        if not selected["input"] and not selected["output"]:
            return {}

    spec: Dict[str, Any] = {}
    for position in ("input", "output"):
        adapter = selected[position]
        if adapter is None:
            continue
        env = dict(adapter.env)
        if model_input:
            env.setdefault("LANDSEER_MODEL_INPUT", model_input)
        if model_path:
            env.setdefault("LANDSEER_MODEL_PATH", model_path)
        entry = {
            "name": adapter.name,
            "image": adapter.container.image,
            "command": adapter.container.command,
            "env": env,
        }
        if position == "output" and model_path:
            entry["model_path"] = model_path
        spec[position] = entry
    return {"adapters": spec}


def adapter_task_config(tool: ToolDefinition) -> Dict[str, Any]:
    """
    Task config entries for a tool's adapters (empty when it has none).

    The adapters are carried in the task config so they are persisted with the
    task, reach the worker unchanged, and are part of the task hash, lineage
    and cache key like the rest of the config. See :func:`resolve_adapters`.
    """
    return resolve_adapters(tool)


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
            model_input=tool_data.get('model_input'),
            model_output=tool_data.get('model_output'),
            model_path=tool_data.get('model_path'),
            adapter_catalog=adapter_defs,
        )
        tool = tools[tool_name]
        for key, allowed in (('model_input', MODEL_INPUT_VALUES), ('model_output', MODEL_OUTPUT_VALUES)):
            value = getattr(tool, key)
            if value is not None and value not in allowed:
                raise ValueError(f"Tool '{tool_name}': {key} {value!r} not in {allowed}")
        if (
            tool.adapters is not None
            and normalize_framework(tool.framework) not in (None, STANDARD_FRAMEWORK)
            and tool.adapters.output is None
            and tool.model_output != 'none'
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
            auto_select=bool(adapter_data.get('auto_select', True)),
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
