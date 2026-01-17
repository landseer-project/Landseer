"""
Landseer Pipeline Module

This module provides the core pipeline infrastructure for the Landseer project,
organized in three layers:

1. Tasks - Individual units of work (pre/in/post/deploy)
2. Workflows - Sequences of tasks representing tool combinations
3. Pipelines - Collections of workflows for ML defense evaluation

Usage:
    from pipeline.config_loader import create_pipeline_from_config
    
    # Load a pipeline from configuration
    pipeline = create_pipeline_from_config("configs/pipeline/trades.yaml")
    
    # Run all workflows
    results = pipeline.run()
    
    # Or run a single workflow
    result = pipeline.run_single_workflow("comb_001")
"""

from .tasks import (
    Task,
    TaskType,
    PreTrainingTask,
    InTrainingTask,
    PostTrainingTask,
    DeploymentTask,
    TaskFactory,
)

from .tools import (
    ToolDefinition,
    Tool,
    ContainerConfig,
    init_tool_registry,
    get_tool,
    get_all_tools,
)

from .workflow import (
    Workflow,
    WorkflowFactory,
)

from .pipeline import (
    Pipeline,
    DefenseEvaluationPipeline,
    PipelineFactory,
)

from .config_loader import (
    PipelineConfig,
    DatasetConfig,
    ModelConfig,
    StageConfig,
    load_pipeline_config,
    create_pipeline_from_config,
    make_combinations,
)

__all__ = [
    # Tasks
    "Task",
    "TaskType",
    "PreTrainingTask",
    "InTrainingTask",
    "PostTrainingTask",
    "DeploymentTask",
    "TaskFactory",
    # Tools
    "ToolDefinition",
    "Tool",
    "ContainerConfig",
    "init_tool_registry",
    "get_tool",
    "get_all_tools",
    # Workflows
    "Workflow",
    "WorkflowFactory",
    # Pipelines
    "Pipeline",
    "DefenseEvaluationPipeline",
    "PipelineFactory",
    # Config loading
    "PipelineConfig",
    "DatasetConfig",
    "ModelConfig",
    "StageConfig",
    "load_pipeline_config",
    "create_pipeline_from_config",
    "make_combinations",
]
