"""
Configuration loader for Landseer pipelines.

This module handles loading and validating pipeline configurations from YAML files,
and creating Pipeline instances with all workflows (combinations).

Workflow generation follows the specification in docs/Tasks.md:
1. Permutations within stages - Order matters: tool1->tool2 != tool2->tool1
2. Single during_training tool - Each workflow can only have 1 during_training tool
3. Baseline substitution - Empty sets should be replaced with baseline tools
4. Task deduplication - Same tool + same dependencies = same task
"""

import os
import shlex
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from itertools import product
from pydantic import BaseModel, Field, field_validator

from .tools import ToolDefinition, Tool, ContainerConfig, get_tool, get_all_tools, init_tool_registry
from .tasks import Task, TaskType, TaskFactory, EvaluationTask, get_or_create_task, clear_task_registry
from .workflow import Workflow, WorkflowFactory
from .pipeline import Pipeline, PipelineFactory
from .workflow_generator import (
    WorkflowGenerator,
    generate_stage_permutations,
    generate_during_training_options,
)
from .stage_validation import load_tools_and_validate_pipeline_stages, resolve_tools_yaml_path
from .container_labels import get_container_labels_for_image

logger = logging.getLogger(__name__)


def normalize_dataset_token(value: str) -> str:
    """Canonical dataset token used for compatibility checks."""
    token = "".join(ch for ch in str(value).lower() if ch.isalnum())
    aliases = {
        "cifar10": "cifar10",
        "cifar100": "cifar100",
        "celeba": "celeba",
        "mnist": "mnist",
    }
    return aliases.get(token, token)


def parse_supported_datasets_label(label_value: str) -> List[str]:
    """Parse org.opencontainers.image.dataset label into canonical dataset tokens."""
    if not label_value:
        return []
    parts = [p.strip() for p in str(label_value).split(",")]
    return [normalize_dataset_token(p) for p in parts if p.strip()]


def validate_pipeline_tool_dataset_compatibility(
    pipeline_config: "PipelineConfig",
    tools_by_id: Dict[str, ToolDefinition],
    requested_dataset_name: str,
    *,
    fetch_remote_labels: bool = True,
) -> List[Dict[str, str]]:
    """
    Validate selected tools support the requested dataset via image labels.

    Returns list of incompatibilities; each item contains stage, tool_id,
    image, requested_dataset, and supported_datasets.
    """
    requested = normalize_dataset_token(requested_dataset_name)
    incompatibilities: List[Dict[str, str]] = []

    # Fallback guardrail for critical tool/dataset constraints when labels are absent.
    strict_dataset_requirements = {
        "in_fair": {"celeba"},
    }

    for stage_name in ("pre_training", "during_training", "post_training", "deployment"):
        stage_cfg = pipeline_config.pipeline.get(stage_name)
        if stage_cfg is None:
            continue
        for tool_entry in getattr(stage_cfg, "tools", []) or []:
            # StageConfig.tools is List[StageToolConfig], but keep backward
            # compatibility with any legacy list[str] callers.
            tool_id = tool_entry.tool if hasattr(tool_entry, "tool") else str(tool_entry)
            tool_def = tools_by_id.get(tool_id)
            if tool_def is None:
                continue
            required_datasets = strict_dataset_requirements.get(tool_id)
            if required_datasets and requested not in required_datasets:
                incompatibilities.append(
                    {
                        "stage": stage_name,
                        "tool_id": tool_id,
                        "tool_name": tool_def.name,
                        "image": tool_def.container.image,
                        "requested_dataset": requested_dataset_name,
                        "supported_datasets": ", ".join(sorted(required_datasets)),
                    }
                )
                continue
            labels: Dict[str, str] = {}
            if fetch_remote_labels:
                labels = get_container_labels_for_image(
                    tool_def.container.image,
                    tool_def.container.runtime,
                )

            supported_raw = labels.get("org.opencontainers.image.dataset", "")
            supported = parse_supported_datasets_label(supported_raw)
            if not supported:
                # Non-breaking behavior: no dataset label means allow.
                continue
            if requested not in supported:
                incompatibilities.append(
                    {
                        "stage": stage_name,
                        "tool_id": tool_id,
                        "tool_name": tool_def.name,
                        "image": tool_def.container.image,
                        "requested_dataset": requested_dataset_name,
                        "supported_datasets": ", ".join(supported),
                    }
                )

    return incompatibilities


# ============================================================================
# Evaluator Configuration
# ============================================================================

class EvaluatorContainerConfig(BaseModel):
    """Container configuration for an evaluator."""
    image: str = Field(description="Container image")
    command: str = Field(description="Command to run")
    runtime: Optional[str] = Field(default=None, description="Container runtime")


class EvaluatorDefinition(BaseModel):
    """Definition of an evaluator from evaluators.yaml."""
    name: str = Field(description="Evaluator name")
    container: EvaluatorContainerConfig = Field(description="Container config")
    required_artifacts: List[str] = Field(default_factory=list, description="Required files")
    metrics: List[str] = Field(default_factory=list, description="Metrics produced")
    defense_types: List[str] = Field(default_factory=list, description="Applicable defense types")
    
    def to_tool(self) -> ToolDefinition:
        """Convert to ToolDefinition for task creation."""
        return ToolDefinition(
            name=self.name,
            container=ContainerConfig(
                image=self.container.image,
                command=self.container.command,
                runtime=self.container.runtime
            ),
            is_baseline=False
        )


# Global evaluator registry
_EVALUATOR_REGISTRY: Dict[str, EvaluatorDefinition] = {}


def _repo_root() -> Path:
    """Project root (directory containing ``configs/``)."""
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(yaml_path: str) -> Path:
    p = Path(yaml_path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return _repo_root() / p


def _builtin_evaluator_definitions() -> Dict[str, EvaluatorDefinition]:
    """
    Default evaluator set — always available so every workflow can attach evaluation tasks.

    Images use ``ghcr.io/landseer-project/evals/<name>:<tag>``; override in ``configs/evaluators.yaml``.
    """
    specs: Dict[str, Dict[str, Any]] = {
        "clean": {
            "name": "clean",
            "container": {"image": "ghcr.io/landseer-project/evals/clean:v2", "command": ""},
            "required_artifacts": [],
            "metrics": ["clean_accuracy"],
            "defense_types": [],
        },
        "backdoor": {
            "name": "backdoor",
            "container": {"image": "ghcr.io/landseer-project/evals/backdoor:v2", "command": ""},
            "required_artifacts": ["poisoning_metadata.json"],
            "metrics": ["attack_success_rate", "clean_accuracy_post_attack", "backdoor_robustness"],
            "defense_types": ["backdoor"],
        },
        "adversarial": {
            "name": "adversarial",
            "container": {"image": "ghcr.io/landseer-project/evals/adversarial:v2", "command": ""},
            "required_artifacts": [],
            "metrics": ["clean_accuracy", "pgd_accuracy", "fgsm_accuracy", "carlini_l2_accuracy"],
            "defense_types": ["adversarial"],
        },
        "fairness": {
            "name": "fairness",
            "container": {"image": "ghcr.io/landseer-project/evals/fairness:v2", "command": ""},
            "required_artifacts": ["sensitive_attributes.npy"],
            "metrics": ["demographic_parity", "equalized_odds_diff"],
            "defense_types": ["fairness"],
        },
        "fingerprinting": {
            "name": "fingerprinting",
            "container": {"image": "ghcr.io/landseer-project/evals/fingerprinting:v2", "command": ""},
            "required_artifacts": [],
            "metrics": ["mingd_score", "fingerprint_accuracy"],
            "defense_types": ["fingerprinting"],
        },
        "ood": {
            "name": "ood",
            "container": {"image": "ghcr.io/landseer-project/evals/ood:v2", "command": ""},
            "required_artifacts": [],
            "metrics": ["ood_auc", "fpr_at_95_tpr"],
            "defense_types": ["outlier_removal"],
        },
        "watermark": {
            "name": "watermark",
            "container": {"image": "ghcr.io/landseer-project/evals/watermark:v2", "command": ""},
            "required_artifacts": ["watermark_key.json"],
            "metrics": ["watermark_accuracy", "bit_accuracy", "detection_rate"],
            "defense_types": ["watermarking"],
        },
    }
    out: Dict[str, EvaluatorDefinition] = {}
    for key, spec in specs.items():
        c = spec["container"]
        out[key] = EvaluatorDefinition(
            name=spec["name"],
            container=EvaluatorContainerConfig(
                image=c["image"],
                command=c.get("command", ""),
                runtime=c.get("runtime"),
            ),
            required_artifacts=spec.get("required_artifacts", []),
            metrics=spec.get("metrics", []),
            defense_types=spec.get("defense_types", []),
        )
    return out


def load_evaluators_from_yaml(yaml_path: str) -> Dict[str, EvaluatorDefinition]:
    """
    Load evaluator definitions from YAML file.
    
    Args:
        yaml_path: Path to evaluators.yaml
        
    Returns:
        Dictionary mapping evaluator names to definitions
    """
    evaluators = {}
    yaml_file = _resolve_config_path(yaml_path)
    
    if not yaml_file.exists():
        logger.debug(f"Evaluators config not found: {yaml_file}")
        return evaluators
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data or 'evaluators' not in data:
        logger.warning(f"No evaluators found in {yaml_path}")
        return evaluators
    
    for eval_name, eval_data in data['evaluators'].items():
        evaluators[eval_name] = EvaluatorDefinition(
            name=eval_data.get('name', eval_name),
            container=EvaluatorContainerConfig(**eval_data['container']),
            required_artifacts=eval_data.get('required_artifacts', []),
            metrics=eval_data.get('metrics', []),
            defense_types=eval_data.get('defense_types', [])
        )
    
    logger.info(f"Loaded {len(evaluators)} evaluators from {yaml_path}")
    return evaluators


def init_evaluator_registry(yaml_path: str = "configs/evaluators.yaml"):
    """
    Initialize the global evaluator registry.

    Built-in evaluators are always registered. If ``configs/evaluators.yaml`` exists,
    its entries override or extend the built-ins by evaluator key.
    """
    global _EVALUATOR_REGISTRY
    base = _builtin_evaluator_definitions()
    path = _resolve_config_path(yaml_path)
    if not path.exists():
        _EVALUATOR_REGISTRY = base.copy()
        logger.info(
            "Using %d built-in evaluators (%s not found; optional overrides there)",
            len(_EVALUATOR_REGISTRY),
            yaml_path,
        )
        return
    try:
        loaded = load_evaluators_from_yaml(yaml_path)
        _EVALUATOR_REGISTRY = {**base, **loaded}
        logger.info(
            "Evaluator registry: %d evaluators (built-ins merged with %s)",
            len(_EVALUATOR_REGISTRY),
            path,
        )
    except Exception as e:
        _EVALUATOR_REGISTRY = base.copy()
        logger.warning(
            "Failed to parse evaluators YAML %s (%s); using %d built-in evaluators only",
            path,
            e,
            len(_EVALUATOR_REGISTRY),
        )


def get_all_evaluators() -> Dict[str, EvaluatorDefinition]:
    """Get all registered evaluators."""
    return _EVALUATOR_REGISTRY.copy()


def load_attack_config(attack_config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load optional attack configuration YAML.

    Returns an empty dict when no path is provided. Raises ValueError for invalid
    files so callers can fail fast with a clear message.
    """
    if not attack_config_path:
        return {}

    path = _resolve_config_path(attack_config_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Attack configuration file not found: {attack_config_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse attack config YAML '{attack_config_path}': {e}")

    if not isinstance(data, dict):
        raise ValueError(
            f"Attack configuration at '{attack_config_path}' must be a mapping/object"
        )

    attacks = data.get("attacks", {})
    if attacks is None:
        attacks = {}
    if not isinstance(attacks, dict):
        raise ValueError(
            f"Attack configuration at '{attack_config_path}' has non-object 'attacks' section"
        )

    normalized_attacks = {
        str(k).strip().lower(): bool(v)
        for k, v in attacks.items()
        if str(k).strip()
    }
    data["attacks"] = normalized_attacks
    data["_resolved_path"] = str(path.resolve())
    return data


def select_evaluators_for_attack_config(
    evaluators: Dict[str, EvaluatorDefinition],
    attack_config: Dict[str, Any],
) -> Dict[str, EvaluatorDefinition]:
    """
    Filter evaluator set based on ``attacks`` flags in attack config.

    ``clean`` is always preserved when present to keep a baseline metric.
    """
    if not attack_config:
        return evaluators

    attacks = attack_config.get("attacks", {})
    if not isinstance(attacks, dict):
        return evaluators

    enabled = {k for k, v in attacks.items() if bool(v)}

    attack_to_evaluator = {
        "backdoor": "backdoor",
        "adversarial": "adversarial",
        "evasion": "adversarial",
        "carlini": "adversarial",
        "fairness": "fairness",
        "fingerprinting": "fingerprinting",
        "watermarking": "watermark",
        "outlier": "ood",
        "ood": "ood",
    }

    selected_keys = {attack_to_evaluator[name] for name in enabled if name in attack_to_evaluator}
    if "clean" in evaluators:
        selected_keys.add("clean")

    selected = {k: v for k, v in evaluators.items() if k in selected_keys}
    if selected:
        return selected

    # If no enabled attack maps to a registered evaluator, keep clean-only when available.
    if "clean" in evaluators:
        return {"clean": evaluators["clean"]}
    return {}


def add_evaluation_tasks_to_workflow(
    workflow: Workflow,
    evaluators: Dict[str, EvaluatorDefinition],
    pipeline_id: str,
    last_deployment_task: Optional[Task] = None,
    attack_config_path: Optional[str] = None,
    attack_config: Optional[Dict[str, Any]] = None,
) -> List[Task]:
    """
    Add evaluation tasks to a workflow.
    
    ALL evaluators are added to every workflow. Evaluators with
    missing required artifacts will skip gracefully at runtime.
    
    Args:
        workflow: Workflow to add tasks to
        evaluators: Dictionary of evaluator definitions
        pipeline_id: Pipeline ID for task tracking
        last_deployment_task: Last task from deployment stage (dependency)
        
    Returns:
        List of created evaluation tasks
    """
    eval_tasks = []

    # Capture non-evaluation defense context once per workflow so evaluators
    # can make model-loading decisions (e.g., Opacus fix for DP checkpoints).
    tools_by_stage: Dict[str, List[str]] = {}
    defense_types: set[str] = set()
    for wf_task in workflow.tasks:
        if wf_task.task_type == TaskType.EVALUATION:
            continue
        stage = wf_task.task_type.value
        tools_by_stage.setdefault(stage, [])
        if wf_task.tool.name not in tools_by_stage[stage]:
            tools_by_stage[stage].append(wf_task.tool.name)
        tool_token = str(wf_task.tool.name).strip().lower().replace("-", "_").replace(" ", "_")
        if tool_token in {"in_dp", "post_dp", "deploy_dp", "dp"} or "differential_privacy" in tool_token:
            defense_types.add("differential_privacy")
    
    for eval_name, eval_def in evaluators.items():
        # Create task config
        task_config = {
            "stage": "evaluation",
            "evaluator": eval_name,
            "tool_name": eval_def.name,
            "metrics": eval_def.metrics,
            "required_artifacts": eval_def.required_artifacts,
            "workflow_tools_by_stage": tools_by_stage,
            "workflow_defense_types": sorted(defense_types),
            "attack_config_path": attack_config_path,
            "attack_flags": (attack_config or {}).get("attacks", {}),
            "attack_config": attack_config or {},
        }
        
        # Create dependencies
        dependencies = [last_deployment_task] if last_deployment_task else []
        
        # Create evaluation task
        task = EvaluationTask(
            tool=eval_def.to_tool(),
            config=task_config,
            priority=50,  # Low priority - runs last
            dependencies=dependencies,
            required_artifacts=eval_def.required_artifacts
        )
        task.pipeline_id = pipeline_id
        
        workflow.add_task(task)
        eval_tasks.append(task)
    
    return eval_tasks


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    name: str = Field(description="Dataset name")
    variant: str = Field(default="clean", description="Dataset variant (clean/poisoned)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Dataset parameters")


class ModelConfig(BaseModel):
    """Model configuration."""
    script: str = Field(description="Path to model configuration script")
    framework: str = Field(default="pytorch", description="ML framework")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")

    @field_validator("script", mode="after")
    def validate_script_exists(cls, v):
        """Validate that the model script exists."""
        v_abs = os.path.abspath(v)
        if not os.path.exists(v_abs):
            logger.warning(f"Model script '{v_abs}' does not exist (validation may be deferred)")
        return v_abs


class StageToolConfig(BaseModel):
    """Pipeline-stage tool selection with optional command override."""
    tool: str = Field(description="Tool registry key or display name")
    command: Optional[str] = Field(
        default=None,
        description="Optional full command override to execute inside the container",
    )
    args: List[str] = Field(
        default_factory=list,
        description="Optional CLI arguments appended to the tool default command",
    )

    @field_validator("tool", mode="after")
    def validate_tool_name(cls, v):
        tool_name = str(v).strip()
        if not tool_name:
            raise ValueError("tool cannot be empty")
        return tool_name

    @field_validator("args", mode="after")
    def validate_args(cls, v):
        out: List[str] = []
        for item in v or []:
            token = str(item).strip()
            if token:
                out.append(token)
        return out

    @property
    def effective_command(self) -> Optional[str]:
        """Return explicit command, if any."""
        cmd = (self.command or "").strip()
        return cmd or None


class StageConfig(BaseModel):
    """Configuration for a pipeline stage.
    
    All tools (including noops) are now in the tools list.
    Noops are identified by their is_baseline=true flag in tools.yaml.
    """
    tools: List[StageToolConfig] = Field(
        default_factory=list,
        description=(
            "Tool selections in this stage (including noops). Each entry can provide "
            "a per-tool command override or extra args."
        ),
    )

    @field_validator("tools", mode="before")
    def normalize_tools(cls, v):
        """Accept legacy list[str] and new list[object] formats."""
        if v is None:
            return []
        normalized = []
        for entry in v:
            if isinstance(entry, str):
                normalized.append({"tool": entry})
                continue
            if isinstance(entry, dict):
                tool_name = entry.get("tool") or entry.get("id") or entry.get("name")
                if not tool_name:
                    raise ValueError(f"Invalid stage tool entry: {entry!r} (missing tool/id/name)")
                args = entry.get("args", [])
                normalized.append(
                    {
                        "tool": tool_name,
                        "command": entry.get("command"),
                        "args": args,
                    }
                )
                continue
            raise ValueError(f"Invalid stage tool entry type: {type(entry)}")
        return normalized


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    dataset: DatasetConfig = Field(description="Dataset configuration")
    model: ModelConfig = Field(description="Model configuration")
    pipeline: Dict[str, StageConfig] = Field(description="Pipeline stages and tools")

    @field_validator("pipeline")
    def validate_stages(cls, v):
        """Validate that all required stages are present."""
        required_stages = {"pre_training", "during_training", "post_training", "deployment"}
        missing = required_stages - set(v.keys())
        if missing:
            raise ValueError(f"Missing required pipeline stages: {missing}")
        return v


def load_pipeline_config(
    config_path: str,
    *,
    tools_yaml_path: str = "configs/tools.yaml",
    validate_tool_stages: bool = True,
    fetch_remote_labels_for_stage_validation: bool = True,
) -> PipelineConfig:
    """
    Load and validate a pipeline configuration from YAML.

    When ``validate_tool_stages`` is True, each tool id must exist in ``tools_yaml_path``,
    and any ``defense_stage`` / ``stage`` entry or OCI image stage label must match the
    pipeline stage the tool is listed under (legacy landseer policy; no id-prefix inference).

    Args:
        config_path: Path to the pipeline YAML file
        tools_yaml_path: Catalog used for stage validation (same file as worker registry)
        validate_tool_stages: If False, skip tool catalog / stage checks
        fetch_remote_labels_for_stage_validation: If False, only YAML ``defense_stage`` /
            ``stage`` is used (no docker/registry label fetch; useful for tests)

    Returns:
        Validated PipelineConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Pipeline configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        config = PipelineConfig.model_validate(data)
        if validate_tool_stages and resolve_tools_yaml_path(tools_yaml_path).exists():
            load_tools_and_validate_pipeline_stages(
                config.pipeline,
                tools_yaml_path=tools_yaml_path,
                fetch_remote_labels=fetch_remote_labels_for_stage_validation,
            )
        elif validate_tool_stages:
            logger.warning(
                "Tools file not found at %s — skipping pipeline stage validation",
                tools_yaml_path,
            )
        logger.info(f"Pipeline configuration loaded successfully from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Failed to validate pipeline configuration: {e}")


def get_stage_tool_definitions(
    stage_config: StageConfig
) -> tuple[List[ToolDefinition], Optional[ToolDefinition]]:
    """
    Get tool definitions from a stage config, separating actual and baseline tools.
    
    Args:
        stage_config: Stage configuration with tool names
        
    Returns:
        Tuple of (actual_tools, baseline_tool)
    """
    actual_tools = []
    baseline_tool = None
    
    # Build a reverse map (display name → ToolDefinition) for fallback lookup.
    # This lets tools_override send either the registry key OR the display name.
    all_tools_dict = get_all_tools()
    name_to_tool: Dict[str, ToolDefinition] = {t.name: t for t in all_tools_dict.values()}

    for tool_cfg in stage_config.tools:
        tool_name = tool_cfg.tool
        tool_def = get_tool(tool_name)
        if tool_def is None:
            # Fallback: try matching by display name (frontend sends this)
            tool_def = name_to_tool.get(tool_name)
        if tool_def is None:
            logger.warning(f"Tool '{tool_name}' not found in registry (tried key and display name), skipping")
            continue

        # Optional stage-level command customization.
        # - command: replaces tool default command entirely
        # - args: appends tokens to the selected base command
        command_override = tool_cfg.effective_command
        if command_override:
            effective_command = command_override
        elif tool_cfg.args:
            effective_command = f"{tool_def.container.command} {' '.join(shlex.quote(arg) for arg in tool_cfg.args)}"
        else:
            effective_command = tool_def.container.command

        if effective_command != tool_def.container.command:
            tool_def = tool_def.model_copy(
                update={
                    "container": tool_def.container.model_copy(
                        update={"command": effective_command}
                    )
                }
            )

        if tool_def.is_baseline:
            baseline_tool = tool_def
        else:
            actual_tools.append(tool_def)
    
    return actual_tools, baseline_tool


def make_combinations(config: PipelineConfig) -> List[Dict[str, List[ToolDefinition]]]:
    """
    Generate all combinations of tools from the pipeline configuration.
    
    This implements the workflow generation logic from docs/Tasks.md:
    1. For pre_training, post_training, deployment: generate all permutations of tool subsets
       - [B, B2] -> [B, B2], [B2, B], [B], [B2], [baseline]
    2. For during_training: single tool only (no permutations)
       - [C, D] -> [C], [D], [baseline]
    3. Combine using Cartesian product
    
    Args:
        config: Validated pipeline configuration
        
    Returns:
        List of dictionaries, each mapping stage names to tool definition lists
    """
    combinations = []
    
    # For each stage, build a list of tool options with proper permutation logic
    stage_options: Dict[str, List[List[ToolDefinition]]] = {}
    
    stages = ["pre_training", "during_training", "post_training", "deployment"]
    
    for stage_name in stages:
        stage_config = config.pipeline.get(stage_name)
        
        if stage_config is None or not stage_config.tools:
            stage_options[stage_name] = [[]]
            continue
        
        actual_tools, baseline_tool = get_stage_tool_definitions(stage_config)
        
        if stage_name == "during_training":
            # Each workflow can only have 1 during_training tool
            options = generate_during_training_options(actual_tools, baseline_tool)
        else:
            # Full permutation for pre_training, post_training, deployment
            options = generate_stage_permutations(actual_tools, baseline_tool)
        
        stage_options[stage_name] = options
    
    # Create Cartesian product of all stage options
    stage_option_lists = [stage_options.get(stage, [[]]) for stage in stages]
    
    for combo_tuple in product(*stage_option_lists):
        combination = {}
        for i, stage in enumerate(stages):
            combination[stage] = combo_tuple[i]
        combinations.append(combination)
    
    logger.info(f"Generated {len(combinations)} workflow combinations from pipeline configuration")
    return combinations


def create_workflow_from_combination(
    combo_id: str,
    combination: Dict[str, List[ToolDefinition]],
    config: PipelineConfig,
    pipeline_id: str = "",
    evaluators: Optional[Dict[str, EvaluatorDefinition]] = None,
    attack_config_path: Optional[str] = None,
    attack_config: Optional[Dict[str, Any]] = None,
) -> Workflow:
    """
    Create a workflow from a tool combination.
    
    Sets up proper dependencies between stages and within stages:
    - pre_training: first tool has no dependencies (priority 100)
    - during_training: depends on last pre_training tool (priority 90)
    - post_training: depends on during_training (priority 80)
    - deployment: depends on post_training (priority 70)
    - evaluation: depends on deployment (priority 50)
    
    Within a stage, tools are chained: tool1 -> tool2 -> tool3
    This implements the ordering requirement from Tasks.md.
    
    Task deduplication is applied: tasks with the same tool, config, and
    dependencies are reused across workflows.
    
    Args:
        combo_id: Unique identifier for this combination
        combination: Dictionary mapping stages to ToolDefinition lists
        config: Pipeline configuration
        pipeline_id: Pipeline ID for task deduplication
        evaluators: Optional dictionary of evaluator definitions
        
    Returns:
        Workflow instance with tasks for each tool
    """
    workflow = WorkflowFactory.create_workflow(
        name=combo_id,
        metadata={
            "dataset": config.dataset.name,
            "model_script": config.model.script
        },
        pipeline_id=pipeline_id
    )
    
    # Map stage names to task types
    stage_to_task_type = {
        "pre_training": TaskType.PRE_TRAINING,
        "during_training": TaskType.IN_TRAINING,
        "post_training": TaskType.POST_TRAINING,
        "deployment": TaskType.DEPLOYMENT,
    }
    
    # Priority based on dependency depth (as per OVERVIEWv1.md)
    # 100 for 0 deps, 90 for 1 dep, 80 for 2 deps, etc.
    stage_priority = {
        "pre_training": 100,      # 0 dependencies
        "during_training": 90,    # 1 dependency (pre_training)
        "post_training": 80,      # 2 dependencies
        "deployment": 70,         # 3 dependencies
    }
    
    # Track tasks from previous stage for dependencies
    previous_stage_tasks: List[Task] = []
    last_deployment_task: Optional[Task] = None
    
    # Create tasks for each stage in order
    for stage_name in ["pre_training", "during_training", "post_training", "deployment"]:
        tool_sequence = combination.get(stage_name, [])
        task_type = stage_to_task_type[stage_name]
        base_priority = stage_priority[stage_name]
        
        current_stage_tasks: List[Task] = []
        
        for idx, tool_def in enumerate(tool_sequence):
            # Determine dependencies based on position
            # First tool in stage depends on previous stage's last task
            # Subsequent tools depend on previous tool in this stage
            if idx == 0:
                dependencies = previous_stage_tasks[-1:] if previous_stage_tasks else []
            else:
                dependencies = [current_stage_tasks[-1]]
            
            # Use get_or_create_task for deduplication across workflows
            task = get_or_create_task(
                task_type=task_type,
                tool=tool_def,
                config={"stage": stage_name, "tool_name": tool_def.name},
                priority=base_priority,
                dependencies=dependencies,
                pipeline_id=pipeline_id
            )
            
            workflow.add_task(task)
            current_stage_tasks.append(task)
        
        # Current stage's last task becomes dependency for next stage
        previous_stage_tasks = current_stage_tasks
        
        # Track last deployment task for evaluation dependencies
        if stage_name == "deployment" and current_stage_tasks:
            last_deployment_task = current_stage_tasks[-1]
    
    # Add evaluation tasks if evaluators are provided
    if evaluators:
        add_evaluation_tasks_to_workflow(
            workflow=workflow,
            evaluators=evaluators,
            pipeline_id=pipeline_id,
            last_deployment_task=last_deployment_task,
            attack_config_path=attack_config_path,
            attack_config=attack_config,
        )
    
    return workflow


def create_pipeline_from_config(
    config_path: str,
    tools_yaml_path: str = "configs/tools.yaml",
    evaluators_yaml_path: str = "configs/evaluators.yaml",
    pipeline_name: Optional[str] = None,
    clear_registry: bool = True,
    include_evaluation: bool = True,
    dataset_name: Optional[str] = None,
    dataset_variant: Optional[str] = None,
    tools_override: Optional[Dict[str, List[str]]] = None,
    model_script: Optional[str] = None,
    attack_config_path: Optional[str] = None,
) -> Pipeline:
    """
    Create a complete Pipeline instance from a configuration file.
    
    This is the main entry point for loading a pipeline. It:
    1. Initializes the tool registry from tools.yaml
    2. Loads evaluators from evaluators.yaml
    3. Loads and validates the pipeline configuration
    4. Generates all tool combinations (with permutations per Tasks.md)
    5. Creates workflows for each combination with task deduplication
    6. Adds evaluation tasks to each workflow
    7. Returns a ready-to-execute Pipeline instance
    
    The workflow generation follows docs/Tasks.md:
    - pre_training, post_training, deployment: all permutations of tool subsets
    - during_training: single tool only
    - Tasks are deduplicated: same tool + same dependencies = same task
    - Evaluation runs after deployment on all workflows
    
    Args:
        config_path: Path to the pipeline YAML configuration file
        tools_yaml_path: Path to the tools.yaml file
        evaluators_yaml_path: Path to the evaluators.yaml file
        pipeline_name: Optional custom pipeline name
        clear_registry: If True, clear task registry before creating (for fresh start)
        include_evaluation: If True, add evaluation tasks to each workflow
        
    Returns:
        Pipeline instance ready for execution
    """
    # Initialize tool registry if not already done
    try:
        init_tool_registry(tools_yaml_path)
    except FileNotFoundError:
        logger.warning(f"Tools config file not found at {tools_yaml_path}, continuing without tool registry")
    
    # Load optional attack configuration once and propagate to evaluator task payloads.
    attack_config = load_attack_config(attack_config_path)

    # Load evaluators if enabled (built-ins always registered; YAML merges on top)
    evaluators = None
    if include_evaluation:
        try:
            init_evaluator_registry(evaluators_yaml_path)
            evaluators = get_all_evaluators()
            if attack_config:
                evaluators = select_evaluators_for_attack_config(evaluators, attack_config)
            logger.info(f"Pipeline will attach {len(evaluators)} evaluator(s) per workflow")
        except Exception as e:
            logger.warning(f"Failed to load evaluators: {e}")
            global _EVALUATOR_REGISTRY
            _EVALUATOR_REGISTRY = _builtin_evaluator_definitions().copy()
            evaluators = get_all_evaluators()
            if attack_config:
                evaluators = select_evaluators_for_attack_config(evaluators, attack_config)
    
    # Clear task registry for a fresh start if requested
    if clear_registry:
        clear_task_registry()

    # Load pipeline configuration (stage validation uses same tools catalog as the worker)
    config = load_pipeline_config(
        config_path,
        tools_yaml_path=tools_yaml_path,
        validate_tool_stages=True,
        fetch_remote_labels_for_stage_validation=True,
    )

    # Apply dataset overrides (e.g. from the UI "Start Run" dialog)
    if dataset_name is not None:
        config.dataset.name = dataset_name
    if dataset_variant is not None:
        config.dataset.variant = dataset_variant

    # Apply model script override
    if model_script is not None:
        config.model.script = model_script

    # Apply tool overrides: replace the tool list for any specified stage.
    # Unknown stage names are ignored to keep this forward-compatible.
    if tools_override:
        for stage_name, tool_names in tools_override.items():
            if stage_name in config.pipeline:
                config.pipeline[stage_name] = StageConfig(
                    tools=[{"tool": tool_name} for tool_name in tool_names]
                )
            else:
                logger.warning(f"tools_override: unknown stage '{stage_name}', skipping")

    # Determine pipeline name
    if pipeline_name is None:
        pipeline_name = Path(config_path).stem

    # Create an empty pipeline first to get its ID
    from .pipeline import DefenseEvaluationPipeline
    pipeline = DefenseEvaluationPipeline(
        name=pipeline_name,
        workflows=[],
        config={"config_path": str(config_path)},
        dataset=config.dataset.model_dump(),
        model=config.model.model_dump()
    )
    
    # Generate combinations with permutation logic
    combinations = make_combinations(config)
    
    # Create workflows from combinations with task deduplication
    # Use the actual pipeline ID for proper task ownership
    for idx, combo in enumerate(combinations, start=1):
        combo_id = f"comb_{idx:03d}"
        workflow = create_workflow_from_combination(
            combo_id, combo, config, 
            pipeline_id=pipeline.id,
            evaluators=evaluators,
            attack_config_path=attack_config_path,
            attack_config=attack_config,
        )
        pipeline.add_workflow(workflow)
    
    # Log summary statistics
    all_tasks = set()
    eval_task_count = 0
    for wf in pipeline.workflows:
        for task in wf.tasks:
            all_tasks.add(task.id)
            if task.task_type == TaskType.EVALUATION:
                eval_task_count += 1
    
    total_task_instances = sum(len(wf.tasks) for wf in pipeline.workflows)
    task_savings = total_task_instances - len(all_tasks)
    
    logger.info(
        f"Created pipeline '{pipeline_name}' with {len(pipeline.workflows)} workflows, "
        f"{len(all_tasks)} unique tasks ({eval_task_count} evaluation tasks), "
        f"saved {task_savings} duplicate executions"
    )
    
    return pipeline


def get_workflow_generation_summary(config_path: str, tools_yaml_path: str = "configs/tools.yaml") -> Dict[str, Any]:
    """
    Get a summary of what workflows would be generated from a config.
    
    This is useful for understanding the combinatorial explosion before
    actually creating the pipeline.
    
    Args:
        config_path: Path to the pipeline YAML configuration file
        tools_yaml_path: Path to the tools.yaml file
        
    Returns:
        Summary dictionary with workflow and task statistics
    """
    # Initialize tool registry
    try:
        init_tool_registry(tools_yaml_path)
    except FileNotFoundError:
        logger.warning(f"Tools config file not found at {tools_yaml_path}")
        return {"error": "Tools config not found"}
    
    # Load pipeline configuration
    config = load_pipeline_config(
        config_path,
        tools_yaml_path=tools_yaml_path,
        validate_tool_stages=True,
        fetch_remote_labels_for_stage_validation=True,
    )

    # Get stage options without creating actual tasks
    stages = ["pre_training", "during_training", "post_training", "deployment"]
    stage_stats = {}
    total_combinations = 1
    
    for stage_name in stages:
        stage_config = config.pipeline.get(stage_name)
        
        if stage_config is None or not stage_config.tools:
            stage_stats[stage_name] = {"options": 1, "tools": 0}
            continue
        
        actual_tools, baseline_tool = get_stage_tool_definitions(stage_config)
        
        if stage_name == "during_training":
            # Single tool options only
            options = generate_during_training_options(actual_tools, baseline_tool)
        else:
            # Full permutations
            options = generate_stage_permutations(actual_tools, baseline_tool)
        
        stage_stats[stage_name] = {
            "options": len(options),
            "actual_tools": len(actual_tools),
            "has_baseline": baseline_tool is not None,
        }
        total_combinations *= len(options)
    
    return {
        "config_path": config_path,
        "total_workflows": total_combinations,
        "stage_stats": stage_stats,
    }
