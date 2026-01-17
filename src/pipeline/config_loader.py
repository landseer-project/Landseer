"""
Configuration loader for Landseer pipelines.

This module handles loading and validating pipeline configurations from YAML files,
and creating Pipeline instances with all workflows (combinations).
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from itertools import product
from pydantic import BaseModel, Field, field_validator

from .tools import ToolDefinition, Tool, get_tool, init_tool_registry
from .tasks import Task, TaskType, TaskFactory
from .workflow import Workflow, WorkflowFactory
from .pipeline import Pipeline, PipelineFactory

logger = logging.getLogger(__name__)


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


class StageConfig(BaseModel):
    """Configuration for a pipeline stage.
    
    All tools (including noops) are now in the tools list.
    Noops are identified by their is_baseline=true flag in tools.yaml.
    """
    tools: List[str] = Field(default_factory=list, description="Tool names in this stage (including noops)")


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


def load_pipeline_config(config_path: str) -> PipelineConfig:
    """
    Load and validate a pipeline configuration from YAML.
    
    Args:
        config_path: Path to the pipeline YAML file
        
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
        logger.info(f"Pipeline configuration loaded successfully from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ValueError(f"Failed to validate pipeline configuration: {e}")


def make_combinations(config: PipelineConfig) -> List[Dict[str, List[str]]]:
    """
    Generate all combinations of tools from the pipeline configuration.
    
    Creates combinations by taking the Cartesian product of tools across all stages.
    Each stage has a list of tools (including noops), and we generate all possible
    combinations of one tool per stage.
    
    Args:
        config: Validated pipeline configuration
        
    Returns:
        List of dictionaries, each mapping stage names to tool name for that combination
    """
    combinations = []
    
    # For each stage, build a list of tool options (all tools including noops)
    stage_options = {}
    
    for stage_name, stage_config in config.pipeline.items():
        # All tools (including noops) are now in the tools list
        options = [[tool] for tool in stage_config.tools] if stage_config.tools else [[]]
        stage_options[stage_name] = options
    
    # Get the ordered list of stages
    stages = ["pre_training", "during_training", "post_training", "deployment"]
    
    # Create cartesian product of all stage options
    stage_option_lists = [stage_options.get(stage, [[]]) for stage in stages]
    
    for combo_tuple in product(*stage_option_lists):
        combination = {}
        for i, stage in enumerate(stages):
            combination[stage] = combo_tuple[i]
        combinations.append(combination)
    
    logger.info(f"Generated {len(combinations)} combinations from pipeline configuration")
    return combinations


def create_workflow_from_combination(
    combo_id: str,
    combination: Dict[str, List[str]],
    config: PipelineConfig
) -> Workflow:
    """
    Create a workflow from a tool combination.
    
    Args:
        combo_id: Unique identifier for this combination
        combination: Dictionary mapping stages to tool name lists
        config: Pipeline configuration
        
    Returns:
        Workflow instance with tasks for each tool
    """
    workflow = WorkflowFactory.create_workflow(
        name=combo_id,
        metadata={
            "dataset": config.dataset.name,
            "model_script": config.model.script
        }
    )
    
    # Map stage names to task types
    stage_to_task_type = {
        "pre_training": TaskType.PRE_TRAINING,
        "during_training": TaskType.IN_TRAINING,
        "post_training": TaskType.POST_TRAINING,
        "deployment": TaskType.DEPLOYMENT,
    }
    
    # Create tasks for each stage in order
    for stage_name in ["pre_training", "during_training", "post_training", "deployment"]:
        tool_names = combination.get(stage_name, [])
        task_type = stage_to_task_type[stage_name]
        
        for tool_name in tool_names:
            # Get tool definition from the global registry
            tool_def = get_tool(tool_name)
            
            if tool_def is None:
                logger.warning(f"Tool '{tool_name}' not found in registry, skipping")
                continue
            
            # Create task directly with tool and config
            task = TaskFactory.create_task(
                task_type=task_type,
                tool=tool_def,
                config={"stage": stage_name},
                priority=0
            )
            
            workflow.add_task(task)
    
    return workflow


def create_pipeline_from_config(
    config_path: str,
    tools_yaml_path: str = "configs/tools.yaml",
    pipeline_name: Optional[str] = None
) -> Pipeline:
    """
    Create a complete Pipeline instance from a configuration file.
    
    This is the main entry point for loading a pipeline. It:
    1. Initializes the tool registry from tools.yaml
    2. Loads and validates the pipeline configuration
    3. Generates all tool combinations
    4. Creates workflows for each combination
    5. Returns a ready-to-execute Pipeline instance
    
    Args:
        config_path: Path to the pipeline YAML configuration file
        tools_yaml_path: Path to the tools.yaml file
        pipeline_name: Optional custom pipeline name
        
    Returns:
        Pipeline instance ready for execution
    """
    # Initialize tool registry if not already done
    try:
        init_tool_registry(tools_yaml_path)
    except FileNotFoundError:
        logger.warning(f"Tools config file not found at {tools_yaml_path}, continuing without tool registry")
    
    # Load pipeline configuration
    config = load_pipeline_config(config_path)
    
    # Generate combinations
    combinations = make_combinations(config)
    
    # Create workflows from combinations
    workflows = []
    for idx, combo in enumerate(combinations, start=1):
        combo_id = f"comb_{idx:03d}"
        workflow = create_workflow_from_combination(combo_id, combo, config)
        workflows.append(workflow)
    
    # Create pipeline
    if pipeline_name is None:
        pipeline_name = Path(config_path).stem
    
    pipeline = PipelineFactory.create_pipeline(
        name=pipeline_name,
        workflows=workflows,
        config={"config_path": str(config_path)},
        dataset=config.dataset.model_dump(),
        model=config.model.model_dump()
    )
    
    logger.info(f"Created pipeline '{pipeline_name}' with {len(workflows)} workflows")
    return pipeline
