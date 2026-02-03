"""
Workflow generator for the Landseer pipeline.

This module handles the generation of all possible workflows from a pipeline configuration
based on the specification in docs/Tasks.md:

1. Permutations within stages - Order matters: tool1->tool2 != tool2->tool1
2. Single during_training tool - Each workflow can only have 1 during_training tool
3. Baseline substitution - Empty sets should be replaced with baseline tools
4. Task deduplication - Same tool + same dependencies = same task
"""

from itertools import permutations, product
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

from .tools import ToolDefinition, get_tool, get_all_tools
from .tasks import (
    Task, TaskType, TaskFactory, get_or_create_task, clear_task_registry
)
from .workflow import Workflow, WorkflowFactory

logger = logging.getLogger(__name__)


@dataclass
class StageTools:
    """
    Container for tools in a pipeline stage.
    
    Attributes:
        stage_name: Name of the stage (pre_training, during_training, etc.)
        actual_tools: List of non-baseline tools
        baseline_tool: The baseline/noop tool for this stage
        task_type: The TaskType for this stage
    """
    stage_name: str
    actual_tools: List[ToolDefinition] = field(default_factory=list)
    baseline_tool: Optional[ToolDefinition] = None
    task_type: TaskType = TaskType.PRE_TRAINING
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all tools including baseline."""
        tools = self.actual_tools.copy()
        if self.baseline_tool:
            tools.append(self.baseline_tool)
        return tools


def generate_stage_permutations(
    tools: List[ToolDefinition],
    baseline: Optional[ToolDefinition] = None,
    include_empty_as_baseline: bool = True
) -> List[List[ToolDefinition]]:
    """
    Generate all permutations for a stage including subsets.
    
    For tools [B, B2], generates:
    - [B, B2], [B2, B]  (all permutations of full set)
    - [B], [B2]         (single tool sets)
    - [baseline]        (empty set -> substitute with baseline)
    
    This implements the permutation logic from Tasks.md:
    "If there was a tool B2 in the pre-training stage, then we would permute B and B2,
    which would give us ordered sets (B, B2), (B2, B), (B), (B2), and a null set."
    
    Args:
        tools: List of actual tools (non-baseline)
        baseline: Baseline tool to use for empty set
        include_empty_as_baseline: If True, include baseline for empty set case
        
    Returns:
        List of tool sequences (permutations)
    """
    result = []
    
    if not tools:
        # No actual tools, just use baseline
        if baseline and include_empty_as_baseline:
            result.append([baseline])
        else:
            result.append([])
        return result
    
    # Generate permutations of all sizes from len(tools) down to 1
    for size in range(len(tools), 0, -1):
        for perm in permutations(tools, size):
            result.append(list(perm))
    
    # Add baseline for the empty set case
    if baseline and include_empty_as_baseline:
        result.append([baseline])
    
    return result


def generate_during_training_options(
    tools: List[ToolDefinition],
    baseline: Optional[ToolDefinition] = None
) -> List[List[ToolDefinition]]:
    """
    Generate options for during_training stage.
    
    Each workflow can only have 1 during_training tool, so we return
    single-tool options only (no permutations of multiple tools).
    
    Args:
        tools: List of tools for during_training
        baseline: Baseline tool
        
    Returns:
        List of single-tool options
    """
    result = []
    
    # Each actual tool as a single option
    for tool in tools:
        result.append([tool])
    
    # Baseline as an option (for when no actual tool is used)
    if baseline:
        result.append([baseline])
    
    # If no tools at all, return empty list of options (but this shouldn't happen)
    if not result:
        result.append([])
    
    return result


class WorkflowGenerator:
    """
    Generates all possible workflows from a pipeline configuration.
    
    This class implements the workflow generation logic from docs/Tasks.md:
    1. Permutes tools within stages (order matters)
    2. Enforces single tool for during_training stage
    3. Substitutes baseline tools for empty sets
    4. Enables task deduplication across workflows
    """
    
    # Stage order for dependency tracking
    STAGE_ORDER = ["pre_training", "during_training", "post_training", "deployment"]
    
    # Map stage names to task types
    STAGE_TO_TASK_TYPE = {
        "pre_training": TaskType.PRE_TRAINING,
        "during_training": TaskType.IN_TRAINING,
        "post_training": TaskType.POST_TRAINING,
        "deployment": TaskType.DEPLOYMENT,
    }
    
    # Priority based on dependency depth
    STAGE_PRIORITY = {
        "pre_training": 100,      # 0 dependencies
        "during_training": 90,    # 1 dependency (pre_training)
        "post_training": 80,      # 2 dependencies
        "deployment": 70,         # 3 dependencies
    }
    
    def __init__(self, pipeline_id: str = ""):
        """
        Initialize the workflow generator.
        
        Args:
            pipeline_id: ID of the pipeline these workflows belong to
        """
        self.pipeline_id = pipeline_id
        self._stage_tools: Dict[str, StageTools] = {}
        self._task_cache: Dict[str, Task] = {}  # hash -> Task for deduplication
    
    def set_stage_tools(
        self,
        stage_name: str,
        actual_tools: List[ToolDefinition],
        baseline_tool: Optional[ToolDefinition] = None
    ) -> None:
        """
        Set the tools for a stage.
        
        Args:
            stage_name: Name of the stage
            actual_tools: List of non-baseline tools
            baseline_tool: The baseline/noop tool
        """
        if stage_name not in self.STAGE_TO_TASK_TYPE:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        self._stage_tools[stage_name] = StageTools(
            stage_name=stage_name,
            actual_tools=actual_tools,
            baseline_tool=baseline_tool,
            task_type=self.STAGE_TO_TASK_TYPE[stage_name]
        )
    
    def generate_all_stage_options(self) -> Dict[str, List[List[ToolDefinition]]]:
        """
        Generate all tool options for each stage.
        
        Returns:
            Dict mapping stage names to lists of tool sequences
        """
        options = {}
        
        for stage_name in self.STAGE_ORDER:
            stage_tools = self._stage_tools.get(stage_name)
            
            if stage_tools is None:
                # No tools for this stage, use empty list
                options[stage_name] = [[]]
                continue
            
            if stage_name == "during_training":
                # Special handling: only single tools allowed
                options[stage_name] = generate_during_training_options(
                    stage_tools.actual_tools,
                    stage_tools.baseline_tool
                )
            else:
                # Full permutation for other stages
                options[stage_name] = generate_stage_permutations(
                    stage_tools.actual_tools,
                    stage_tools.baseline_tool
                )
        
        return options
    
    def _get_or_create_deduplicated_task(
        self,
        tool: ToolDefinition,
        task_type: TaskType,
        priority: int,
        config: Dict[str, Any],
        dependencies: List[Task]
    ) -> Task:
        """
        Get an existing task or create a new one with deduplication.
        
        Tasks with the same tool, config, and dependencies are reused.
        
        Args:
            tool: Tool definition
            task_type: Type of task
            priority: Task priority
            config: Task configuration
            dependencies: List of dependency tasks
            
        Returns:
            Task instance (new or existing)
        """
        return get_or_create_task(
            task_type=task_type,
            tool=tool,
            config=config,
            priority=priority,
            dependencies=dependencies,
            pipeline_id=self.pipeline_id
        )
    
    def _create_tasks_for_stage_sequence(
        self,
        tool_sequence: List[ToolDefinition],
        task_type: TaskType,
        base_priority: int,
        previous_stage_tasks: List[Task],
        stage_name: str
    ) -> List[Task]:
        """
        Create tasks for a sequence of tools in a stage.
        
        Tasks within the same stage are chained: first tool has no stage dependencies,
        second tool depends on first, etc. All depend on the previous stage's final task.
        
        Args:
            tool_sequence: Ordered list of tools to execute
            task_type: TaskType for this stage
            base_priority: Base priority for this stage
            previous_stage_tasks: Tasks from the previous stage (for dependencies)
            stage_name: Name of the stage
            
        Returns:
            List of tasks for this stage
        """
        stage_tasks = []
        
        for idx, tool in enumerate(tool_sequence):
            # Determine dependencies
            if idx == 0:
                # First tool in stage depends on previous stage's last task
                dependencies = previous_stage_tasks[-1:] if previous_stage_tasks else []
            else:
                # Subsequent tools depend on previous tool in this stage
                dependencies = [stage_tasks[-1]]
            
            # Create task with deduplication
            task = self._get_or_create_deduplicated_task(
                tool=tool,
                task_type=task_type,
                priority=base_priority,
                config={"stage": stage_name, "tool_name": tool.name},
                dependencies=dependencies
            )
            
            stage_tasks.append(task)
        
        return stage_tasks
    
    def generate_workflow(
        self,
        combination: Dict[str, List[ToolDefinition]],
        workflow_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Workflow:
        """
        Create a workflow from a tool combination.
        
        Args:
            combination: Dict mapping stage names to tool sequences
            workflow_name: Name for the workflow
            metadata: Optional metadata
            
        Returns:
            Workflow instance
        """
        workflow = WorkflowFactory.create_workflow(
            name=workflow_name,
            metadata=metadata or {},
            pipeline_id=self.pipeline_id
        )
        
        previous_stage_tasks: List[Task] = []
        
        for stage_name in self.STAGE_ORDER:
            tool_sequence = combination.get(stage_name, [])
            
            if not tool_sequence:
                continue
            
            task_type = self.STAGE_TO_TASK_TYPE[stage_name]
            base_priority = self.STAGE_PRIORITY[stage_name]
            
            stage_tasks = self._create_tasks_for_stage_sequence(
                tool_sequence=tool_sequence,
                task_type=task_type,
                base_priority=base_priority,
                previous_stage_tasks=previous_stage_tasks,
                stage_name=stage_name
            )
            
            for task in stage_tasks:
                workflow.add_task(task)
            
            previous_stage_tasks = stage_tasks
        
        return workflow
    
    def generate_all_workflows(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Workflow]:
        """
        Generate all possible workflows from the configured stages.
        
        This creates the Cartesian product of all stage options, where:
        - pre_training, post_training, deployment: all permutations of tool subsets
        - during_training: single tool options only
        
        Args:
            metadata: Optional metadata to add to all workflows
            
        Returns:
            List of all generated workflows
        """
        stage_options = self.generate_all_stage_options()
        
        workflows = []
        workflow_idx = 1
        
        # Generate Cartesian product of all stage options
        option_lists = [stage_options[stage] for stage in self.STAGE_ORDER]
        
        for combo_tuple in product(*option_lists):
            combination = {}
            for i, stage in enumerate(self.STAGE_ORDER):
                combination[stage] = combo_tuple[i]
            
            workflow_name = f"comb_{workflow_idx:03d}"
            workflow = self.generate_workflow(
                combination=combination,
                workflow_name=workflow_name,
                metadata=metadata
            )
            
            workflows.append(workflow)
            workflow_idx += 1
        
        logger.info(f"Generated {len(workflows)} workflows with task deduplication")
        
        # Log deduplication stats
        all_tasks = set()
        for wf in workflows:
            for task in wf.tasks:
                all_tasks.add(task.id)
        
        logger.info(f"Total unique tasks: {len(all_tasks)}")
        
        return workflows
    
    def get_workflow_summary(self, workflows: List[Workflow]) -> Dict[str, Any]:
        """
        Get a summary of the generated workflows.
        
        Args:
            workflows: List of workflows
            
        Returns:
            Summary statistics dictionary
        """
        all_tasks: Set[str] = set()
        task_usage_count: Dict[str, int] = {}
        
        for wf in workflows:
            for task in wf.tasks:
                all_tasks.add(task.id)
                task_usage_count[task.id] = task_usage_count.get(task.id, 0) + 1
        
        # Find most reused tasks
        max_reuse = max(task_usage_count.values()) if task_usage_count else 0
        
        return {
            "total_workflows": len(workflows),
            "total_unique_tasks": len(all_tasks),
            "total_task_instances": sum(len(wf.tasks) for wf in workflows),
            "task_reuse_savings": sum(len(wf.tasks) for wf in workflows) - len(all_tasks),
            "max_task_reuse": max_reuse,
        }


def create_workflows_from_stage_config(
    stage_config: Dict[str, Any],
    tool_registry: Dict[str, ToolDefinition],
    pipeline_id: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> List[Workflow]:
    """
    Create workflows from a stage configuration dictionary.
    
    This is a convenience function that wraps WorkflowGenerator.
    
    Args:
        stage_config: Dict mapping stage names to lists of tool names
        tool_registry: Dict of tool name -> ToolDefinition
        pipeline_id: Pipeline ID for task deduplication
        metadata: Optional metadata for workflows
        
    Returns:
        List of generated workflows
    """
    generator = WorkflowGenerator(pipeline_id=pipeline_id)
    
    for stage_name, tool_names in stage_config.items():
        if stage_name not in generator.STAGE_ORDER:
            continue
        
        actual_tools = []
        baseline_tool = None
        
        for tool_name in tool_names:
            tool_def = tool_registry.get(tool_name)
            if tool_def is None:
                logger.warning(f"Tool '{tool_name}' not found in registry, skipping")
                continue
            
            if tool_def.is_baseline:
                baseline_tool = tool_def
            else:
                actual_tools.append(tool_def)
        
        generator.set_stage_tools(
            stage_name=stage_name,
            actual_tools=actual_tools,
            baseline_tool=baseline_tool
        )
    
    return generator.generate_all_workflows(metadata=metadata)
