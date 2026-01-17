"""
Example usage and tests for the Landseer scheduler system.

This script demonstrates:
1. Creating tasks with dependencies
2. Task deduplication based on hash
3. Building workflows and pipelines
4. Using the PriorityScheduler
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pipeline.tasks import (
    Task, TaskType, TaskStatus, TaskFactory,
    get_or_create_task, clear_task_registry
)
from pipeline.tools import ToolDefinition, ContainerConfig
from pipeline.workflow import Workflow, WorkflowFactory
from pipeline.pipeline import Pipeline, DefenseEvaluationPipeline, PipelineFactory
from backend.scheduler.base_scheduler import Scheduler
from backend.scheduler.priority_scheduler import PriorityScheduler


def create_sample_tools():
    """Create sample tool definitions."""
    tools = {
        "outlier_detection": ToolDefinition(
            name="outlier_detection",
            container=ContainerConfig(
                image="ghcr.io/landseer/pre_outlier:v1",
                command="python main.py"
            )
        ),
        "data_augmentation": ToolDefinition(
            name="data_augmentation",
            container=ContainerConfig(
                image="ghcr.io/landseer/pre_augment:v1",
                command="python augment.py"
            )
        ),
        "adversarial_training": ToolDefinition(
            name="adversarial_training",
            container=ContainerConfig(
                image="ghcr.io/landseer/in_advtrain:v1",
                command="python train.py"
            )
        ),
        "fine_pruning": ToolDefinition(
            name="fine_pruning",
            container=ContainerConfig(
                image="ghcr.io/landseer/post_prune:v1",
                command="python prune.py"
            )
        ),
        "magnet": ToolDefinition(
            name="magnet",
            container=ContainerConfig(
                image="ghcr.io/landseer/post_magnet:v2",
                command="python main.py"
            )
        ),
    }
    return tools


def example_basic_scheduler():
    """Example 1: Basic scheduler usage with simple workflow."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Scheduler Usage")
    print("=" * 70)
    
    # Clear registry for clean start
    clear_task_registry()
    
    tools = create_sample_tools()
    
    # Create a simple workflow with dependencies
    # Task 1: outlier detection (no dependencies)
    task1 = TaskFactory.create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=tools["outlier_detection"],
        config={"threshold": 0.95}
    )
    
    # Task 2: adversarial training (depends on task1)
    task2 = TaskFactory.create_task(
        task_type=TaskType.IN_TRAINING,
        tool=tools["adversarial_training"],
        config={"epsilon": 0.3},
        dependencies=[task1]
    )
    
    # Task 3: fine pruning (depends on task2)
    task3 = TaskFactory.create_task(
        task_type=TaskType.POST_TRAINING,
        tool=tools["fine_pruning"],
        config={"prune_rate": 0.1},
        dependencies=[task2]
    )
    
    # Create workflow
    workflow = WorkflowFactory.create_workflow(
        name="simple_defense",
        tasks=[task1, task2, task3]
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(
        name="test_pipeline",
        workflows=[workflow]
    )
    
    # Create scheduler
    scheduler = PriorityScheduler(pipeline)
    
    print(f"\nCreated pipeline '{pipeline.name}' with ID: {pipeline.id}")
    print(f"Workflow '{workflow.name}' with ID: {workflow.id}")
    print(f"Total tasks: {len(scheduler.get_all_tasks())}")
    
    # Show initial priorities
    print("\nInitial task priorities:")
    for task in scheduler.get_all_tasks():
        info = scheduler.get_task_priority_info(task.id)
        print(f"  {task.id}: priority={info['priority']}, "
              f"level={info['dependency_level']}, counter={info['usage_counter']}")
    
    # Simulate execution
    print("\nSimulating execution:")
    step = 1
    while not scheduler.is_complete():
        next_task = scheduler.get_next_task()
        
        if next_task is None:
            print("  No tasks ready (checking for deadlock...)")
            break
        
        print(f"\n  Step {step}: Executing {next_task.id} "
              f"(tool: {next_task.tool.name}, priority: {next_task.priority})")
        
        # Simulate task completion
        scheduler.update_task_status(next_task.id, TaskStatus.COMPLETED)
        print(f"    ✓ Task {next_task.id} completed")
        
        # Show progress
        progress = scheduler.get_progress()
        print(f"    Progress: {progress['completed']}/{progress['total']} completed")
        
        step += 1
    
    print("\n✓ All tasks completed!")
    print("=" * 70)


def example_task_deduplication():
    """Example 2: Task deduplication with shared tasks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Task Deduplication")
    print("=" * 70)
    
    # Clear registry
    clear_task_registry()
    
    tools = create_sample_tools()
    
    # Create a pipeline
    pipeline = PipelineFactory.create_pipeline(name="dedup_test")
    
    # Workflow 1: outlier -> adversarial training -> fine pruning
    task1_w1 = get_or_create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=tools["outlier_detection"],
        config={"threshold": 0.95},
        pipeline_id=pipeline.id
    )
    
    task2_w1 = get_or_create_task(
        task_type=TaskType.IN_TRAINING,
        tool=tools["adversarial_training"],
        config={"epsilon": 0.3},
        dependencies=[task1_w1],
        pipeline_id=pipeline.id
    )
    
    task3_w1 = get_or_create_task(
        task_type=TaskType.POST_TRAINING,
        tool=tools["fine_pruning"],
        config={"prune_rate": 0.1},
        dependencies=[task2_w1],
        pipeline_id=pipeline.id
    )
    
    workflow1 = WorkflowFactory.create_workflow(
        name="workflow_1",
        tasks=[task1_w1, task2_w1, task3_w1],
        pipeline_id=pipeline.id
    )
    
    # Workflow 2: outlier -> adversarial training -> magnet
    # The first two tasks should be REUSED!
    task1_w2 = get_or_create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=tools["outlier_detection"],
        config={"threshold": 0.95},  # Same config as workflow 1
        pipeline_id=pipeline.id
    )
    
    task2_w2 = get_or_create_task(
        task_type=TaskType.IN_TRAINING,
        tool=tools["adversarial_training"],
        config={"epsilon": 0.3},  # Same config as workflow 1
        dependencies=[task1_w2],
        pipeline_id=pipeline.id
    )
    
    task3_w2 = get_or_create_task(
        task_type=TaskType.POST_TRAINING,
        tool=tools["magnet"],
        config={},
        dependencies=[task2_w2],
        pipeline_id=pipeline.id
    )
    
    workflow2 = WorkflowFactory.create_workflow(
        name="workflow_2",
        tasks=[task1_w2, task2_w2, task3_w2],
        pipeline_id=pipeline.id
    )
    
    # Add workflows to pipeline
    pipeline.add_workflow(workflow1)
    pipeline.add_workflow(workflow2)
    
    # Verify deduplication
    print(f"\nCreated 2 workflows with 3 tasks each")
    print(f"Total unique task instances created: {len([task1_w1, task2_w1, task3_w1, task1_w2, task2_w2, task3_w2])}")
    print(f"\nTask reuse verification:")
    print(f"  task1_w1 is task1_w2: {task1_w1 is task1_w2} (should be True)")
    print(f"  task2_w1 is task2_w2: {task2_w1 is task2_w2} (should be True)")
    print(f"  task3_w1 is task3_w2: {task3_w1 is task3_w2} (should be False - different tools)")
    
    # Show task counters
    print(f"\nTask usage counters:")
    print(f"  {task1_w1.id} (outlier): counter={task1_w1.counter}, workflows={task1_w1.workflows}")
    print(f"  {task2_w1.id} (advtrain): counter={task2_w1.counter}, workflows={task2_w1.workflows}")
    print(f"  {task3_w1.id} (prune): counter={task3_w1.counter}, workflows={task3_w1.workflows}")
    print(f"  {task3_w2.id} (magnet): counter={task3_w2.counter}, workflows={task3_w2.workflows}")
    
    # Create scheduler
    scheduler = PriorityScheduler(pipeline)
    
    print(f"\nScheduler manages {len(scheduler.get_all_tasks())} unique tasks")
    
    # Show priority levels
    levels = scheduler.get_priority_levels()
    print(f"\nTasks by dependency level:")
    for level, tasks in sorted(levels.items()):
        print(f"  Level {level} ({len(tasks)} tasks):")
        for task in tasks:
            print(f"    - {task.id}: tool={task.tool.name}, "
                  f"counter={task.counter}, priority={task.priority}")
    
    print("=" * 70)


def example_complex_pipeline():
    """Example 3: Complex pipeline with multiple workflows and priorities."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Complex Pipeline with Priority Scheduling")
    print("=" * 70)
    
    # Clear registry
    clear_task_registry()
    
    tools = create_sample_tools()
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(
        name="complex_defense_eval",
        dataset={"name": "cifar10", "variant": "clean"},
        model={"framework": "pytorch", "script": "models/resnet.py"}
    )
    
    # Create 3 different workflows with overlapping tasks
    
    # Workflow 1: Just outlier detection
    t1 = get_or_create_task(
        TaskType.PRE_TRAINING,
        tools["outlier_detection"],
        config={"threshold": 0.95},
        pipeline_id=pipeline.id
    )
    
    workflow1 = WorkflowFactory.create_workflow(
        "baseline",
        tasks=[t1],
        pipeline_id=pipeline.id
    )
    
    # Workflow 2: outlier + adversarial training
    t2 = get_or_create_task(
        TaskType.IN_TRAINING,
        tools["adversarial_training"],
        config={"epsilon": 0.3},
        dependencies=[t1],
        pipeline_id=pipeline.id
    )
    
    workflow2 = WorkflowFactory.create_workflow(
        "defense_1",
        tasks=[t1, t2],
        pipeline_id=pipeline.id
    )
    
    # Workflow 3: outlier + adversarial + fine pruning
    t3 = get_or_create_task(
        TaskType.POST_TRAINING,
        tools["fine_pruning"],
        config={"prune_rate": 0.1},
        dependencies=[t2],
        pipeline_id=pipeline.id
    )
    
    workflow3 = WorkflowFactory.create_workflow(
        "defense_2",
        tasks=[t1, t2, t3],
        pipeline_id=pipeline.id
    )
    
    # Workflow 4: outlier + adversarial + magnet
    t4 = get_or_create_task(
        TaskType.POST_TRAINING,
        tools["magnet"],
        config={},
        dependencies=[t2],
        pipeline_id=pipeline.id
    )
    
    workflow4 = WorkflowFactory.create_workflow(
        "defense_3",
        tasks=[t1, t2, t4],
        pipeline_id=pipeline.id
    )
    
    # Add all workflows
    pipeline.add_workflow(workflow1)
    pipeline.add_workflow(workflow2)
    pipeline.add_workflow(workflow3)
    pipeline.add_workflow(workflow4)
    
    print(f"\nPipeline: {pipeline.name} (ID: {pipeline.id})")
    print(f"Dataset: {pipeline.dataset['name']}")
    print(f"Workflows: {len(pipeline.workflows)}")
    
    # Create scheduler
    scheduler = PriorityScheduler(pipeline)
    
    print(f"\nUnique tasks in scheduler: {len(scheduler.get_all_tasks())}")
    
    # Show detailed priority information
    print("\n" + "-" * 70)
    print("Task Priority Analysis:")
    print("-" * 70)
    
    for task in sorted(scheduler.get_all_tasks(), key=lambda t: t.priority):
        info = scheduler.get_task_priority_info(task.id)
        print(f"\n{task.id}:")
        print(f"  Tool: {task.tool.name}")
        print(f"  Priority: {info['priority']} "
              f"(level={info['dependency_level']}, counter={info['usage_counter']})")
        print(f"  Used in {info['usage_counter']} workflow(s): {', '.join(info['workflows'])}")
        print(f"  Dependencies: {len(info['dependencies'])} "
              f"({', '.join(info['dependencies']) if info['dependencies'] else 'none'})")
    
    # Simulate execution with detailed output
    print("\n" + "-" * 70)
    print("Execution Simulation:")
    print("-" * 70)
    
    step = 1
    while not scheduler.is_complete():
        # Show ready tasks
        ready_tasks = scheduler.get_ready_tasks_by_priority()
        
        if not ready_tasks:
            print("\nNo tasks ready!")
            break
        
        print(f"\n--- Step {step} ---")
        print(f"Ready tasks: {len(ready_tasks)}")
        for i, task in enumerate(ready_tasks[:3], 1):  # Show top 3
            print(f"  {i}. {task.id} (priority={task.priority}, tool={task.tool.name})")
        
        # Get and execute next task
        next_task = scheduler.get_next_task()
        print(f"\n→ Selected: {next_task.id}")
        print(f"  Tool: {next_task.tool.name}")
        print(f"  Priority: {next_task.priority}")
        print(f"  Counter: {next_task.counter}")
        
        # Complete the task
        scheduler.update_task_status(next_task.id, TaskStatus.COMPLETED)
        
        # Show progress
        progress = scheduler.get_progress()
        print(f"\n  Progress: {progress['completed']}/{progress['total']} completed, "
              f"{progress['pending']} pending, {progress['running']} running")
        
        step += 1
    
    print("\n✓ Pipeline execution complete!")
    
    final_progress = scheduler.get_progress()
    print(f"\nFinal statistics:")
    print(f"  Total tasks: {final_progress['total']}")
    print(f"  Completed: {final_progress['completed']}")
    print(f"  Failed: {final_progress['failed']}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run all examples
    example_basic_scheduler()
    example_task_deduplication()
    example_complex_pipeline()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully! ✓")
    print("=" * 70)
