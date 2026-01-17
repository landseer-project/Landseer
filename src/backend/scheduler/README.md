# Landseer Scheduler System

This module provides a flexible scheduler system for managing task execution in the Landseer ML defense pipeline.

## Features

- **Task Management**: Tasks with unique IDs, dependencies, priorities, and status tracking
- **Task Deduplication**: Automatically reuses tasks with identical tool, config, and dependencies
- **Priority-Based Scheduling**: Prioritizes tasks based on dependency level and usage frequency
- **Status Tracking**: Track task execution status (pending, running, completed, failed)
- **Workflow Organization**: Group tasks into workflows representing tool combinations
- **Pipeline Management**: Manage multiple workflows in a single pipeline

## Architecture

### Core Components

1. **Task** (`pipeline/tasks.py`)
   - Base unit of work with tool, config, and dependencies
   - Hashable for deduplication
   - Tracks status, priority, usage counter, and workflows
   - Stable unique IDs (task_1, task_2, etc.)

2. **Workflow** (`pipeline/workflow.py`)
   - Ordered sequence of tasks
   - Represents one tool combination
   - Unique IDs (workflow_1, workflow_2, etc.)

3. **Pipeline** (`pipeline/pipeline.py`)
   - Container for multiple workflows
   - Manages dataset and model configuration
   - Unique IDs (pipeline_1, pipeline_2, etc.)

4. **Scheduler** (`backend/scheduler/base_scheduler.py`)
   - Abstract base class for schedulers
   - Provides `get_next_task()` and `update_task_status()` interface
   - Tracks task progress and completion

5. **PriorityScheduler** (`backend/scheduler/priority_scheduler.py`)
   - Concrete scheduler implementation
   - Priority formula: `(dependency_level * 1000) - usage_counter`
   - Lower number = higher priority

## Task Status Lifecycle

```
PENDING → RUNNING → COMPLETED
                  ↓
                FAILED
```

- **PENDING**: Initial state, waiting for dependencies
- **RUNNING**: Currently being executed
- **COMPLETED**: Successfully finished
- **FAILED**: Execution failed

## Task Deduplication

Tasks are deduplicated based on a hash of:
- Tool name and container configuration
- Task configuration parameters
- Dependency list

When creating tasks with `get_or_create_task()`, identical tasks are reused across workflows within the same pipeline.

## Usage Examples

### Basic Scheduler Usage

```python
from pipeline.tasks import TaskFactory, TaskType
from pipeline.tools import ToolDefinition, ContainerConfig
from pipeline.workflow import WorkflowFactory
from pipeline.pipeline import PipelineFactory
from backend.scheduler import PriorityScheduler

# Create tools
tool = ToolDefinition(
    name="outlier_detection",
    container=ContainerConfig(
        image="ghcr.io/landseer/pre_outlier:v1",
        command="python main.py"
    )
)

# Create tasks with dependencies
task1 = TaskFactory.create_task(
    task_type=TaskType.PRE_TRAINING,
    tool=tool,
    config={"threshold": 0.95}
)

task2 = TaskFactory.create_task(
    task_type=TaskType.IN_TRAINING,
    tool=another_tool,
    config={"epsilon": 0.3},
    dependencies=[task1]  # Depends on task1
)

# Create workflow and pipeline
workflow = WorkflowFactory.create_workflow(
    name="defense_combo",
    tasks=[task1, task2]
)

pipeline = PipelineFactory.create_pipeline(
    name="eval_pipeline",
    workflows=[workflow]
)

# Create scheduler
scheduler = PriorityScheduler(pipeline)

# Execute tasks
while not scheduler.is_complete():
    task = scheduler.get_next_task()
    if task is None:
        break
    
    # Execute the task
    result = execute_task(task)
    
    # Update status
    status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
    scheduler.update_task_status(task.id, status)
```

### Task Deduplication Example

```python
from pipeline.tasks import get_or_create_task, TaskType

# Create pipeline
pipeline = PipelineFactory.create_pipeline(name="test")

# Workflow 1: task A → task B
task_a_w1 = get_or_create_task(
    TaskType.PRE_TRAINING,
    tool_a,
    config={"param": 1},
    pipeline_id=pipeline.id
)

task_b_w1 = get_or_create_task(
    TaskType.IN_TRAINING,
    tool_b,
    config={"param": 2},
    dependencies=[task_a_w1],
    pipeline_id=pipeline.id
)

# Workflow 2: task A → task C
# task_a_w2 will be THE SAME INSTANCE as task_a_w1!
task_a_w2 = get_or_create_task(
    TaskType.PRE_TRAINING,
    tool_a,
    config={"param": 1},  # Same config
    pipeline_id=pipeline.id
)

# Verify: task_a_w1 is task_a_w2 == True
# Counter: task_a_w1.counter == 2 (used in 2 workflows)
```

### Priority Scheduling

Priority is calculated as:
```
priority = (dependency_level * 1000) - usage_counter
```

Examples:
- Task with 0 dependencies, used in 5 workflows: priority = 0 - 5 = **-5** (highest)
- Task with 0 dependencies, used in 2 workflows: priority = 0 - 2 = **-2**
- Task with 1 dependency, used in 3 workflows: priority = 1000 - 3 = **997**
- Task with 2 dependencies, used in 1 workflow: priority = 2000 - 1 = **1999** (lowest)

Within each dependency level, tasks with higher usage get priority.

## API Reference

### Scheduler Methods

#### `get_next_task() -> Optional[Task]`
Returns the next task to execute, or None if no tasks are ready.
Automatically updates task status to RUNNING.

#### `update_task_status(task_id: str, status: TaskStatus) -> None`
Updates the status of a completed task. Status must be COMPLETED or FAILED.

#### `is_complete() -> bool`
Returns True if all tasks have completed (successfully or failed).

#### `get_progress() -> dict`
Returns progress statistics:
```python
{
    "total": 10,
    "pending": 2,
    "running": 1,
    "completed": 6,
    "failed": 1
}
```

### PriorityScheduler Methods

#### `get_ready_tasks_by_priority() -> List[Task]`
Returns all ready tasks sorted by priority.

#### `get_task_priority_info(task_id: str) -> dict`
Returns detailed priority information for a task:
```python
{
    "task_id": "task_1",
    "priority": -5,
    "dependency_level": 0,
    "usage_counter": 5,
    "status": "pending",
    "dependencies": [],
    "workflows": ["workflow_1", "workflow_2", ...]
}
```

#### `get_priority_levels() -> dict`
Returns tasks grouped by dependency level.

## Running Examples

Run the example script to see the scheduler in action:

```bash
cd /share/landseer/workspace-ayushi/Landseer
python src/backend/scheduler/example_usage.py
```

This will demonstrate:
1. Basic scheduler usage with a simple workflow
2. Task deduplication across multiple workflows
3. Complex pipeline with priority-based scheduling

## Implementation Notes

### ID Generation

IDs are generated using global counters:
- Tasks: `task_1`, `task_2`, ...
- Tools: `tool_1`, `tool_2`, ...
- Workflows: `workflow_1`, `workflow_2`, ...
- Pipelines: `pipeline_1`, `pipeline_2`, ...

### Task Registry

A global task registry (`_task_registry`) stores all created tasks for deduplication.
Use `clear_task_registry()` to reset (useful for testing).

### Thread Safety

The current implementation is **not thread-safe**. If you need concurrent access,
add appropriate locking mechanisms around:
- Task registry access
- Scheduler state updates
- ID counter increments

## Future Enhancements

Potential improvements:
- [ ] Persistent task storage (database)
- [ ] Distributed scheduling across multiple workers
- [ ] Resource-aware scheduling (GPU/CPU allocation)
- [ ] Task retry logic with backoff
- [ ] Real-time monitoring dashboard
- [ ] Task cancellation support
- [ ] Checkpoint/resume functionality

## Integration with Existing Code

To integrate with the existing Landseer pipeline:

1. Replace manual task ordering with scheduler
2. Use `get_or_create_task()` when building workflows
3. Let scheduler manage execution order
4. Update task status after tool execution

Example integration in `PipelineExecutor`:

```python
# Instead of:
for combination in combinations:
    for stage in stages:
        for tool in stage.tools:
            run_tool(tool)

# Use:
scheduler = PriorityScheduler(pipeline)
while not scheduler.is_complete():
    task = scheduler.get_next_task()
    if task:
        result = run_tool(task.tool, task.config)
        status = TaskStatus.COMPLETED if result else TaskStatus.FAILED
        scheduler.update_task_status(task.id, status)
```
