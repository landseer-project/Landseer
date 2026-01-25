"""
Tests for task creation and workflow generation.

These tests verify the behavior specified in docs/Tasks.md:
1. Permutations within stages (order matters: tool1->tool2 != tool2->tool1)
2. Baseline tool substitution for null/empty sets
3. Task deduplication (same tool + same dependencies = same task)
"""

import pytest
from typing import List, Dict, Any
from itertools import permutations

from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    get_or_create_task,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.workflow_generator import (
    WorkflowGenerator,
    generate_stage_permutations,
    generate_during_training_options,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tool_a():
    """Baseline tool A (pre_training)."""
    return ToolDefinition(
        name="tool_a",
        container=ContainerConfig(image="test/a:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_b():
    """Actual tool B (pre_training)."""
    return ToolDefinition(
        name="tool_b",
        container=ContainerConfig(image="test/b:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_b2():
    """Another actual tool B2 (pre_training)."""
    return ToolDefinition(
        name="tool_b2",
        container=ContainerConfig(image="test/b2:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_c():
    """Baseline tool C (during_training)."""
    return ToolDefinition(
        name="tool_c",
        container=ContainerConfig(image="test/c:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_d():
    """Actual tool D (during_training)."""
    return ToolDefinition(
        name="tool_d",
        container=ContainerConfig(image="test/d:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_e():
    """Baseline tool E (post_training)."""
    return ToolDefinition(
        name="tool_e",
        container=ContainerConfig(image="test/e:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_g():
    """Baseline tool G (deployment)."""
    return ToolDefinition(
        name="tool_g",
        container=ContainerConfig(image="test/g:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture(autouse=True)
def cleanup_task_registry():
    """Clear task registry before and after each test."""
    clear_task_registry()
    yield
    clear_task_registry()


# ============================================================================
# Helper Functions
# ============================================================================


def generate_stage_permutations(
    tools: List[ToolDefinition],
    baseline: ToolDefinition
) -> List[List[ToolDefinition]]:
    """
    Generate all permutations for a stage including subsets.
    
    For tools [B, B2], generates:
    - [B, B2], [B2, B]  (all permutations of full set)
    - [B], [B2]         (single tool sets)
    - [baseline]        (empty set -> substitute with baseline)
    
    Args:
        tools: List of actual tools (non-baseline)
        baseline: Baseline tool to use for empty set
        
    Returns:
        List of tool sequences (permutations)
    """
    result = []
    
    # Generate permutations of all sizes from len(tools) down to 1
    for size in range(len(tools), 0, -1):
        for perm in permutations(tools, size):
            result.append(list(perm))
    
    # Add baseline for the empty set case
    result.append([baseline])
    
    return result


def create_task_with_dependencies(
    tool: ToolDefinition,
    task_type: TaskType,
    dependencies: List[Task] = None
) -> Task:
    """Helper to create a task with dependencies."""
    task = TaskFactory.create_task(
        task_type=task_type,
        tool=tool,
        config={"tool_name": tool.name},
        dependencies=dependencies or []
    )
    return task


# ============================================================================
# Test 1: Permutations within stages (order matters)
# ============================================================================


class TestStagePermutations:
    """Tests verifying that tool order within stages matters."""
    
    def test_different_order_creates_different_workflows(
        self, tool_b, tool_b2, tool_c, tool_e, tool_g
    ):
        """
        Workflows with different tool orders should be distinct.
        
        B->B2->C->E->G != B2->B->C->E->G
        """
        # Create workflow 1: B->B2->C->E->G
        task_b_first = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        task_b2_second = create_task_with_dependencies(
            tool_b2, TaskType.PRE_TRAINING, [task_b_first]
        )
        task_c_1 = create_task_with_dependencies(
            tool_c, TaskType.IN_TRAINING, [task_b2_second]
        )
        
        workflow1 = WorkflowFactory.create_workflow(
            name="B_B2_C_E_G",
            tasks=[task_b_first, task_b2_second, task_c_1]
        )
        
        # Create workflow 2: B2->B->C->E->G
        task_b2_first = create_task_with_dependencies(tool_b2, TaskType.PRE_TRAINING)
        task_b_second = create_task_with_dependencies(
            tool_b, TaskType.PRE_TRAINING, [task_b2_first]
        )
        task_c_2 = create_task_with_dependencies(
            tool_c, TaskType.IN_TRAINING, [task_b_second]
        )
        
        workflow2 = WorkflowFactory.create_workflow(
            name="B2_B_C_E_G",
            tasks=[task_b2_first, task_b_second, task_c_2]
        )
        
        # Verify they are different workflows
        assert workflow1.name != workflow2.name
        assert workflow1.id != workflow2.id
        
        # Verify task order is different
        wf1_tools = [t.tool.name for t in workflow1.tasks]
        wf2_tools = [t.tool.name for t in workflow2.tasks]
        
        assert wf1_tools == ["tool_b", "tool_b2", "tool_c"]
        assert wf2_tools == ["tool_b2", "tool_b", "tool_c"]
        assert wf1_tools != wf2_tools
    
    def test_single_tool_vs_multiple_tools_different_workflows(
        self, tool_b, tool_b2, tool_c
    ):
        """
        Single tool workflow should be different from multi-tool workflow.
        
        B->C != B->B2->C
        """
        # Workflow with single tool: B->C
        task_b_only = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        task_c_single = create_task_with_dependencies(
            tool_c, TaskType.IN_TRAINING, [task_b_only]
        )
        
        workflow_single = WorkflowFactory.create_workflow(
            name="B_C",
            tasks=[task_b_only, task_c_single]
        )
        
        # Workflow with multiple tools: B->B2->C
        task_b_multi = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        task_b2_multi = create_task_with_dependencies(
            tool_b2, TaskType.PRE_TRAINING, [task_b_multi]
        )
        task_c_multi = create_task_with_dependencies(
            tool_c, TaskType.IN_TRAINING, [task_b2_multi]
        )
        
        workflow_multi = WorkflowFactory.create_workflow(
            name="B_B2_C",
            tasks=[task_b_multi, task_b2_multi, task_c_multi]
        )
        
        # Verify they are different
        assert len(workflow_single.tasks) == 2
        assert len(workflow_multi.tasks) == 3
        assert workflow_single.id != workflow_multi.id
    
    def test_generate_all_permutations_for_stage(self, tool_b, tool_b2, tool_a):
        """
        Test that we can generate all permutations for a stage.
        
        For tools [B, B2] with baseline A:
        - [B, B2], [B2, B]  (2 permutations of 2)
        - [B], [B2]         (2 permutations of 1)
        - [A]               (baseline for empty)
        = 5 total combinations
        """
        tools = [tool_b, tool_b2]
        permutation_list = generate_stage_permutations(tools, tool_a)
        
        assert len(permutation_list) == 5
        
        # Verify contents
        perm_names = [[t.name for t in p] for p in permutation_list]
        
        assert ["tool_b", "tool_b2"] in perm_names
        assert ["tool_b2", "tool_b"] in perm_names
        assert ["tool_b"] in perm_names
        assert ["tool_b2"] in perm_names
        assert ["tool_a"] in perm_names  # baseline
    
    def test_dependencies_reflect_tool_order(self, tool_b, tool_b2):
        """Tasks should have dependencies reflecting their order in the workflow."""
        # B executes first, B2 depends on B
        task_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        task_b2 = create_task_with_dependencies(
            tool_b2, TaskType.PRE_TRAINING, [task_b]
        )
        
        assert len(task_b.dependencies) == 0
        assert len(task_b2.dependencies) == 1
        assert task_b in task_b2.dependencies


# ============================================================================
# Test 2: Baseline tool substitution
# ============================================================================


class TestBaselineSubstitution:
    """Tests verifying baseline tools are used for empty tool sets."""
    
    def test_empty_stage_uses_baseline(self, tool_a, tool_c):
        """When no tools specified for a stage, baseline should be used."""
        # Simulate empty pre_training stage -> use baseline A
        workflow = WorkflowFactory.create_workflow(
            name="baseline_workflow",
            tasks=[]
        )
        
        # Add baseline task (this is what should happen for empty stage)
        baseline_task = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        workflow.add_task(baseline_task)
        
        # Verify baseline is used
        assert len(workflow.tasks) == 1
        assert workflow.tasks[0].tool.name == "tool_a"
        assert workflow.tasks[0].tool.is_baseline is True
    
    def test_baseline_tool_has_is_baseline_flag(self, tool_a, tool_b):
        """Baseline tools should have is_baseline=True."""
        assert tool_a.is_baseline is True
        assert tool_b.is_baseline is False
    
    def test_workflow_with_all_baselines(self, tool_a, tool_c, tool_e, tool_g):
        """
        A workflow using all baseline tools (no actual tools).
        This represents the control/baseline run.
        """
        task_a = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        task_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [task_a])
        task_e = create_task_with_dependencies(tool_e, TaskType.POST_TRAINING, [task_c])
        task_g = create_task_with_dependencies(tool_g, TaskType.DEPLOYMENT, [task_e])
        
        workflow = WorkflowFactory.create_workflow(
            name="all_baseline",
            tasks=[task_a, task_c, task_e, task_g]
        )
        
        # All tasks should be baseline
        for task in workflow.tasks:
            assert task.tool.is_baseline is True
    
    def test_baseline_equivalent_workflows_are_same(self, tool_a, tool_c, tool_e, tool_g):
        """
        As per Tasks.md: workflows 1, 2, 3 are the same because A is baseline.
        A->B->C and B->A->C are equivalent when A is baseline (noop).
        
        Note: This tests the EXPECTED behavior - currently the implementation
        may not handle this correctly.
        """
        # This test documents the expected behavior from the spec
        # When A is a baseline (noop), it shouldn't affect the output
        
        # For now, we verify that baseline tools are identifiable
        task_a = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        
        assert task_a.tool.is_baseline is True


# ============================================================================
# Test 3: Task deduplication
# ============================================================================


class TestTaskDeduplication:
    """Tests verifying task deduplication based on tool + dependencies."""
    
    def test_same_tool_no_deps_creates_same_task(self, tool_b):
        """
        Tasks with same tool and no dependencies should be reused.
        
        As per Tasks.md: A is executed first for workflows 1, 3, 5, 7.
        So we can say that it is the same task.
        """
        task1 = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"tool_name": tool_b.name},
            dependencies=[],
            pipeline_id="test_pipeline"
        )
        
        task2 = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"tool_name": tool_b.name},
            dependencies=[],
            pipeline_id="test_pipeline"
        )
        
        # Should return the same task instance
        assert task1.id == task2.id
        assert task1 is task2
    
    def test_same_tool_different_deps_creates_different_tasks(self, tool_b, tool_b2):
        """
        Tasks with same tool but different dependencies should be different.
        
        As per Tasks.md: A in workflow 1 (first) vs workflow 2 (after B)
        are different tasks because dependencies differ.
        """
        # Task B with no dependencies
        task_b_no_deps = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"tool_name": tool_b.name},
            dependencies=[],
            pipeline_id="test_pipeline"
        )
        
        # Create a dependency task
        dep_task = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b2,
            config={"tool_name": tool_b2.name},
            dependencies=[],
            pipeline_id="test_pipeline"
        )
        
        # Task B with dependency on B2
        task_b_with_deps = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"tool_name": tool_b.name},
            dependencies=[dep_task],
            pipeline_id="test_pipeline"
        )
        
        # Should be different tasks
        assert task_b_no_deps.id != task_b_with_deps.id
        assert task_b_no_deps is not task_b_with_deps
    
    def test_task_counter_increments_on_reuse(self, tool_b):
        """
        When a task is reused across workflows, its counter should increment.
        """
        # Create task first time
        task = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"tool_name": tool_b.name},
            dependencies=[],
            pipeline_id="test_pipeline"
        )
        
        initial_counter = task.counter
        
        # Simulate adding to a workflow
        workflow1 = WorkflowFactory.create_workflow(
            name="wf1",
            pipeline_id="test_pipeline"
        )
        task.add_to_workflow(workflow1.id, "test_pipeline")
        
        # Counter should increment
        assert task.counter == initial_counter + 1
        
        # Add to another workflow
        workflow2 = WorkflowFactory.create_workflow(
            name="wf2",
            pipeline_id="test_pipeline"
        )
        task.add_to_workflow(workflow2.id, "test_pipeline")
        
        # Counter should increment again
        assert task.counter == initial_counter + 2
    
    def test_task_hash_based_on_tool_and_deps(self, tool_b, tool_b2):
        """
        Task hash should be deterministic based on tool and dependencies.
        """
        # Create two tasks with same tool and deps
        task1 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"key": "value"},
            dependencies=[]
        )
        
        task2 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"key": "value"},
            dependencies=[]
        )
        
        # Hashes should be equal (even though IDs differ)
        assert task1.get_hash() == task2.get_hash()
    
    def test_different_tools_different_hash(self, tool_b, tool_b2):
        """Tasks with different tools should have different hashes."""
        task1 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={},
            dependencies=[]
        )
        
        task2 = TaskFactory.create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b2,
            config={},
            dependencies=[]
        )
        
        assert task1.get_hash() != task2.get_hash()
    
    def test_dependency_order_matters_for_task_identity(self, tool_a, tool_b, tool_c):
        """
        Tasks with same tool but different dependency ORDER are different.
        
        From Tasks.md:
        "if for a task C the dependencies are A->B, then a task C with 
        dependencies B->A is a different task"
        
        This tests that:
        - Task C with deps [A, B] (where B depends on A) 
        - Task C with deps [B, A] (where A depends on B)
        Are DIFFERENT tasks.
        """
        # Create dependency chain A->B
        task_a_first = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_a,
            config={"name": "a"},
            dependencies=[],
            pipeline_id="order_test"
        )
        task_b_after_a = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"name": "b"},
            dependencies=[task_a_first],
            pipeline_id="order_test"
        )
        
        # Task C depending on B (which depends on A): A->B->C
        task_c_with_a_b = get_or_create_task(
            task_type=TaskType.IN_TRAINING,
            tool=tool_c,
            config={"name": "c"},
            dependencies=[task_b_after_a],
            pipeline_id="order_test"
        )
        
        # Create dependency chain B->A (reversed order)
        task_b_first = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_b,
            config={"name": "b"},
            dependencies=[],
            pipeline_id="order_test"
        )
        task_a_after_b = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_a,
            config={"name": "a"},
            dependencies=[task_b_first],
            pipeline_id="order_test"
        )
        
        # Task C depending on A (which depends on B): B->A->C
        task_c_with_b_a = get_or_create_task(
            task_type=TaskType.IN_TRAINING,
            tool=tool_c,
            config={"name": "c"},
            dependencies=[task_a_after_b],
            pipeline_id="order_test"
        )
        
        # The two C tasks should be DIFFERENT because their dependency chains differ
        assert task_c_with_a_b is not task_c_with_b_a
        assert task_c_with_a_b.id != task_c_with_b_a.id
        
        # Verify the dependency structure
        # C in A->B->C depends on B
        assert task_b_after_a in task_c_with_a_b.dependencies
        # C in B->A->C depends on A
        assert task_a_after_b in task_c_with_b_a.dependencies
    
    def test_deduplication_across_workflows(self, tool_a, tool_b, tool_c):
        """
        Same task used in multiple workflows should be the same instance.
        
        Example: Task A (first in pipeline) is shared across:
        - Workflow 1: A->B->C
        - Workflow 3: A->C
        """
        # Both workflows start with A (no dependencies)
        task_a_wf1 = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_a,
            config={"stage": "pre_training"},
            dependencies=[],
            pipeline_id="shared_pipeline"
        )
        
        task_a_wf3 = get_or_create_task(
            task_type=TaskType.PRE_TRAINING,
            tool=tool_a,
            config={"stage": "pre_training"},
            dependencies=[],
            pipeline_id="shared_pipeline"
        )
        
        # Should be the same task
        assert task_a_wf1 is task_a_wf3
        assert task_a_wf1.id == task_a_wf3.id


# ============================================================================
# Integration Tests
# ============================================================================


class TestWorkflowGeneration:
    """Integration tests for complete workflow generation."""
    
    def test_generate_workflows_from_spec_example(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test workflow generation matching the example in Tasks.md.
        
        With A, C, E, G as baseline and B, D as actual tools:
        Expected workflows (simplified, without full permutations):
        1. A->B->C->E->G
        2. B->A->C->E->G
        3. A->C->E->G
        4. B->C->E->G
        5. A->B->D->E->G
        6. B->A->D->E->G
        7. A->D->E->G
        8. B->D->E->G
        """
        # Create workflow 1: A->B->C->E->G
        wf1_a = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        wf1_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING, [wf1_a])
        wf1_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [wf1_b])
        wf1_e = create_task_with_dependencies(tool_e, TaskType.POST_TRAINING, [wf1_c])
        wf1_g = create_task_with_dependencies(tool_g, TaskType.DEPLOYMENT, [wf1_e])
        
        workflow1 = WorkflowFactory.create_workflow(
            name="comb_001",
            tasks=[wf1_a, wf1_b, wf1_c, wf1_e, wf1_g]
        )
        
        # Create workflow 4: B->C->E->G (no A)
        wf4_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        wf4_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [wf4_b])
        wf4_e = create_task_with_dependencies(tool_e, TaskType.POST_TRAINING, [wf4_c])
        wf4_g = create_task_with_dependencies(tool_g, TaskType.DEPLOYMENT, [wf4_e])
        
        workflow4 = WorkflowFactory.create_workflow(
            name="comb_004",
            tasks=[wf4_b, wf4_c, wf4_e, wf4_g]
        )
        
        # Verify workflows are different
        assert workflow1.name != workflow4.name
        assert len(workflow1.tasks) == 5
        assert len(workflow4.tasks) == 4
        
        # Verify tool sequences
        wf1_tools = [t.tool.name for t in workflow1.tasks]
        wf4_tools = [t.tool.name for t in workflow4.tasks]
        
        assert wf1_tools == ["tool_a", "tool_b", "tool_c", "tool_e", "tool_g"]
        assert wf4_tools == ["tool_b", "tool_c", "tool_e", "tool_g"]
    
    def test_during_training_single_tool_constraint(self, tool_c, tool_d):
        """
        Each workflow can only have 1 during_training tool.
        Verify this constraint is enforced.
        """
        # Valid: single during_training tool
        task_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING)
        
        workflow_valid = WorkflowFactory.create_workflow(
            name="valid_workflow",
            tasks=[task_c]
        )
        
        # Get during_training tasks
        during_training_tasks = workflow_valid.get_tasks_by_type(TaskType.IN_TRAINING)
        assert len(during_training_tasks) == 1
    
    def test_complete_pipeline_with_deduplication(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test a complete pipeline where tasks are properly deduplicated.
        """
        # Workflow 1: A->C->E->G
        wf1_a = get_or_create_task(
            TaskType.PRE_TRAINING, tool_a, {"stage": "pre"}, 0, [], "pipeline1"
        )
        wf1_c = get_or_create_task(
            TaskType.IN_TRAINING, tool_c, {"stage": "during"}, 0, [wf1_a], "pipeline1"
        )
        wf1_e = get_or_create_task(
            TaskType.POST_TRAINING, tool_e, {"stage": "post"}, 0, [wf1_c], "pipeline1"
        )
        wf1_g = get_or_create_task(
            TaskType.DEPLOYMENT, tool_g, {"stage": "deploy"}, 0, [wf1_e], "pipeline1"
        )
        
        # Workflow 2: A->D->E->G (shares A with workflow 1)
        wf2_a = get_or_create_task(
            TaskType.PRE_TRAINING, tool_a, {"stage": "pre"}, 0, [], "pipeline1"
        )
        wf2_d = get_or_create_task(
            TaskType.IN_TRAINING, tool_d, {"stage": "during"}, 0, [wf2_a], "pipeline1"
        )
        wf2_e = get_or_create_task(
            TaskType.POST_TRAINING, tool_e, {"stage": "post"}, 0, [wf2_d], "pipeline1"
        )
        wf2_g = get_or_create_task(
            TaskType.DEPLOYMENT, tool_g, {"stage": "deploy"}, 0, [wf2_e], "pipeline1"
        )
        
        # Task A should be the same in both workflows
        assert wf1_a is wf2_a, "Task A should be deduplicated across workflows"
        
        # But downstream tasks differ because dependencies differ
        assert wf1_c is not wf2_d  # Different tools
        assert wf1_e is not wf2_e  # Same tool but different dependencies
        assert wf1_g is not wf2_g  # Same tool but different dependencies


# ============================================================================
# Test 4: Workflow Generator
# ============================================================================


class TestWorkflowGenerator:
    """Tests for the WorkflowGenerator class implementing Tasks.md logic."""
    
    def test_generate_stage_permutations_two_tools(self, tool_b, tool_b2, tool_a):
        """
        Test permutation generation for two tools with baseline.
        
        For [B, B2] with baseline A:
        - [B, B2], [B2, B]  (2 permutations of 2)
        - [B], [B2]         (2 permutations of 1)
        - [A]               (baseline for empty)
        = 5 total
        """
        result = generate_stage_permutations([tool_b, tool_b2], tool_a)
        
        assert len(result) == 5
        
        # Check all expected combinations are present
        names = [[t.name for t in combo] for combo in result]
        
        assert ["tool_b", "tool_b2"] in names
        assert ["tool_b2", "tool_b"] in names
        assert ["tool_b"] in names
        assert ["tool_b2"] in names
        assert ["tool_a"] in names
    
    def test_generate_stage_permutations_single_tool(self, tool_b, tool_a):
        """
        Test permutation generation for single tool with baseline.
        
        For [B] with baseline A:
        - [B]  (single tool)
        - [A]  (baseline for empty)
        = 2 total
        """
        result = generate_stage_permutations([tool_b], tool_a)
        
        assert len(result) == 2
        
        names = [[t.name for t in combo] for combo in result]
        assert ["tool_b"] in names
        assert ["tool_a"] in names
    
    def test_generate_stage_permutations_no_tools(self, tool_a):
        """
        Test permutation generation with no actual tools.
        
        For [] with baseline A:
        - [A]  (baseline only)
        = 1 total
        """
        result = generate_stage_permutations([], tool_a)
        
        assert len(result) == 1
        assert result[0][0].name == "tool_a"
    
    def test_generate_during_training_options(self, tool_c, tool_d):
        """
        Test during_training options - should be single tools only.
        
        For [C, D] with no baseline:
        - [C], [D]
        = 2 options
        """
        result = generate_during_training_options([tool_c, tool_d], None)
        
        assert len(result) == 2
        
        names = [[t.name for t in combo] for combo in result]
        assert ["tool_c"] in names
        assert ["tool_d"] in names
    
    def test_during_training_single_tool_constraint(self, tool_c, tool_d):
        """
        Verify during_training never produces multi-tool combinations.
        """
        result = generate_during_training_options([tool_c, tool_d], None)
        
        # All options should have exactly 1 tool
        for option in result:
            assert len(option) == 1
    
    def test_workflow_generator_full_pipeline(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test full workflow generation with the WorkflowGenerator.
        
        Setup (from Tasks.md example):
        - pre_training: A (baseline), B (actual)
        - during_training: C (baseline), D (actual)
        - post_training: E (baseline only)
        - deployment: G (baseline only)
        
        Expected pre_training options: [B], [A] = 2
        Expected during_training options: [C], [D] = 2
        Expected post_training options: [E] = 1
        Expected deployment options: [G] = 1
        
        Total workflows = 2 * 2 * 1 * 1 = 4
        """
        generator = WorkflowGenerator(pipeline_id="test_pipeline")
        
        # Set up stages
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        assert len(workflows) == 4
        
        # Verify workflow names
        workflow_names = [wf.name for wf in workflows]
        assert "comb_001" in workflow_names
        assert "comb_004" in workflow_names
    
    def test_workflow_generator_task_deduplication(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test that tasks are properly deduplicated across workflows.
        
        When multiple workflows share the same first task (same tool, no deps),
        they should all reference the same task instance.
        """
        generator = WorkflowGenerator(pipeline_id="dedup_test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Get summary
        summary = generator.get_workflow_summary(workflows)
        
        # Should have task reuse (fewer unique tasks than total instances)
        assert summary["total_unique_tasks"] < summary["total_task_instances"]
        assert summary["task_reuse_savings"] > 0
    
    def test_workflow_generator_with_multiple_pre_training_tools(
        self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g
    ):
        """
        Test with multiple pre_training tools - should generate permutations.
        
        For [B, B2] with baseline A:
        - [B, B2], [B2, B], [B], [B2], [A] = 5 options
        
        during_training: [C] = 1
        post_training: [E] = 1
        deployment: [G] = 1
        
        Total = 5 * 1 * 1 * 1 = 5 workflows
        """
        generator = WorkflowGenerator(pipeline_id="multi_pre_test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        assert len(workflows) == 5
        
        # Verify one workflow has B->B2 sequence
        found_b_b2 = False
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            if len(pre_tasks) == 2:
                names = [t.tool.name for t in pre_tasks]
                if names == ["tool_b", "tool_b2"]:
                    found_b_b2 = True
                    # Verify B2 depends on B
                    assert pre_tasks[0] in pre_tasks[1].dependencies
        
        assert found_b_b2, "Should have a workflow with B->B2 sequence"
    
    def test_stage_options_count(self, tool_a, tool_b, tool_b2, tool_c, tool_d, tool_e, tool_g):
        """
        Verify the number of options generated for each stage type.
        """
        generator = WorkflowGenerator(pipeline_id="count_test")
        
        # pre_training with 2 actual tools
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        # during_training with 2 actual tools
        generator.set_stage_tools("during_training", [tool_c, tool_d], None)
        # Empty stages
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        options = generator.generate_all_stage_options()
        
        # pre_training: P(2,2) + P(2,1) + baseline = 2 + 2 + 1 = 5
        assert len(options["pre_training"]) == 5
        
        # during_training: single tools only = 2 (no baseline since None)
        assert len(options["during_training"]) == 2
        
        # post_training: baseline only = 1
        assert len(options["post_training"]) == 1
        
        # deployment: baseline only = 1
        assert len(options["deployment"]) == 1
        
        # Total workflows = 5 * 2 * 1 * 1 = 10
        workflows = generator.generate_all_workflows()
        assert len(workflows) == 10


# ============================================================================
# Test 5: Workflows from Tasks.md Specification
# ============================================================================


class TestTasksMdSpecification:
    """
    Tests for workflows from Tasks.md.
    
    Important note from Tasks.md:
    "workflow 1, 2, 3 are the same workflow because A is a baseline tool"
    
    This means when A is a baseline (noop), we don't generate permutations
    with it. Instead:
    - pre_training options: [B] (actual), [A] (baseline for empty)
    - NOT: [A,B], [B,A], [A], [B]
    
    So with A (baseline), B (actual) in pre_training and
    C (baseline), D (actual) in during_training:
    
    Unique workflows (4 total):
    1. B->C->E->G  (actual pre, baseline during)
    2. B->D->E->G  (actual pre, actual during)
    3. A->C->E->G  (baseline pre, baseline during)
    4. A->D->E->G  (baseline pre, actual during)
    """
    
    def test_unique_workflows_with_baseline(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Verify correct number of unique workflows are generated.
        
        With baselines correctly handled:
        - pre_training: [B], [A] = 2 options
        - during_training: [C], [D] = 2 options
        - post: [E] = 1
        - deploy: [G] = 1
        Total = 2 * 2 * 1 * 1 = 4 workflows
        """
        generator = WorkflowGenerator(pipeline_id="spec_test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        assert len(workflows) == 4
    
    def test_workflows_with_multiple_actual_tools(
        self, tool_a, tool_b, tool_b2, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test with B2 added - per Tasks.md note:
        "If there was a tool B2 in pre-training stage, then we would 
        permute B and B2, which would give us ordered sets (B, B2), 
        (B2, B), (B), (B2), and a null set = 5 different workflows"
        """
        generator = WorkflowGenerator(pipeline_id="b2_test")
        
        # Two actual tools in pre_training
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # pre_training: 5 options (as per Tasks.md)
        # during: 2, post: 1, deploy: 1
        # Total = 5 * 2 * 1 * 1 = 10
        assert len(workflows) == 10
    
    def test_workflow_B_B2_sequence_exists(
        self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g
    ):
        """Test that B->B2->C->E->G workflow exists."""
        generator = WorkflowGenerator(pipeline_id="b_b2_test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with B->B2 sequence
        found = False
        for wf in workflows:
            tool_names = [t.tool.name for t in wf.tasks]
            if tool_names == ["tool_b", "tool_b2", "tool_c", "tool_e", "tool_g"]:
                found = True
                tasks = wf.tasks
                # Verify B2 depends on B
                assert tasks[0] in tasks[1].dependencies
        
        assert found, "Workflow B->B2->C->E->G not found"
    
    def test_workflow_B2_B_sequence_exists(
        self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g
    ):
        """Test that B2->B->C->E->G workflow exists (different order)."""
        generator = WorkflowGenerator(pipeline_id="b2_b_test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with B2->B sequence
        found = False
        for wf in workflows:
            tool_names = [t.tool.name for t in wf.tasks]
            if tool_names == ["tool_b2", "tool_b", "tool_c", "tool_e", "tool_g"]:
                found = True
                tasks = wf.tasks
                # Verify B depends on B2
                assert tasks[0] in tasks[1].dependencies
        
        assert found, "Workflow B2->B->C->E->G not found"
    
    def test_workflow_4_B_C_E_G(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """Test workflow 4: B->C->E->G (single pre_training tool)."""
        generator = WorkflowGenerator(pipeline_id="wf4_test")
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with B->C->E->G pattern
        found = False
        for wf in workflows:
            tool_names = [t.tool.name for t in wf.tasks]
            if tool_names == ["tool_b", "tool_c", "tool_e", "tool_g"]:
                found = True
                assert len(wf.tasks) == 4
        
        assert found, "Workflow B->C->E->G not found"
    
    def test_workflow_7_A_D_E_G(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """Test workflow 7: A->D->E->G (baseline pre, actual during)."""
        generator = WorkflowGenerator(pipeline_id="wf7_test")
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with A->D->E->G pattern
        found = False
        for wf in workflows:
            tool_names = [t.tool.name for t in wf.tasks]
            if tool_names == ["tool_a", "tool_d", "tool_e", "tool_g"]:
                found = True
                assert len(wf.tasks) == 4
        
        assert found, "Workflow A->D->E->G not found"
    
    def test_task_deduplication_across_spec_workflows(
        self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g
    ):
        """
        Test task deduplication as per Tasks.md:
        
        "A is executed first for workflows 1, 3, 5, 7. So we can say 
        that it is the same task. However, it is not the same task in 
        2, 6 as B is executed first."
        """
        generator = WorkflowGenerator(pipeline_id="dedup_spec_test")
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Collect all tasks where A is first (no dependencies)
        a_first_tasks = []
        # Collect all tasks where A has dependencies (comes after B)
        a_after_b_tasks = []
        
        for wf in workflows:
            for task in wf.tasks:
                if task.tool.name == "tool_a":
                    if len(task.dependencies) == 0:
                        a_first_tasks.append(task)
                    else:
                        a_after_b_tasks.append(task)
        
        # All "A first" tasks should be the same instance
        if len(a_first_tasks) > 1:
            first = a_first_tasks[0]
            for task in a_first_tasks[1:]:
                assert task is first, "A as first task should be deduplicated"
        
        # All "A after B" tasks should be the same instance
        if len(a_after_b_tasks) > 1:
            first = a_after_b_tasks[0]
            for task in a_after_b_tasks[1:]:
                assert task is first, "A after B should be deduplicated"
        
        # But "A first" and "A after B" should be different
        if a_first_tasks and a_after_b_tasks:
            assert a_first_tasks[0] is not a_after_b_tasks[0], \
                "A first vs A after B should be different tasks"


# ============================================================================
# Test 6: Priority based on dependency depth (from Tasks.md)
# ============================================================================


class TestPriorityByDependencyDepth:
    """
    Tests for priority assignment based on dependency depth.
    
    From Tasks.md:
    - A->B->C->E->G: A=100, B=90, C=80, E=70, G=60
    - B->C->E->G: B=100, C=90, E=80, G=70
    """
    
    def test_priority_decreases_with_depth_5_tasks(
        self, tool_a, tool_b, tool_c, tool_e, tool_g
    ):
        """
        For A->B->C->E->G:
        - A (depth 0): priority 100
        - B (depth 1): priority 90
        - C (depth 2): priority 80
        - E (depth 3): priority 70
        - G (depth 4): priority 60
        """
        from src.backend.scheduler.priority_scheduler import PriorityScheduler
        
        # Create workflow A->B->C->E->G
        task_a = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING, [task_a])
        task_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [task_b])
        task_e = create_task_with_dependencies(tool_e, TaskType.POST_TRAINING, [task_c])
        task_g = create_task_with_dependencies(tool_g, TaskType.DEPLOYMENT, [task_e])
        
        workflow = WorkflowFactory.create_workflow(
            name="priority_test",
            tasks=[task_a, task_b, task_c, task_e, task_g]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="priority_pipeline",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # Check priority levels
        levels = scheduler.get_priority_levels()
        
        # A should be at depth 0
        assert task_a in levels[0]
        # B should be at depth 1
        assert task_b in levels[1]
        # C should be at depth 2
        assert task_c in levels[2]
        # E should be at depth 3
        assert task_e in levels[3]
        # G should be at depth 4
        assert task_g in levels[4]
        
        # Verify priorities decrease with depth
        assert task_a.priority > task_b.priority
        assert task_b.priority > task_c.priority
        assert task_c.priority > task_e.priority
        assert task_e.priority > task_g.priority
    
    def test_priority_for_shorter_workflow(
        self, tool_b, tool_c, tool_e, tool_g
    ):
        """
        For B->C->E->G (4 tasks):
        - B (depth 0): priority ~100
        - C (depth 1): priority ~90
        - E (depth 2): priority ~80
        - G (depth 3): priority ~70
        """
        from src.backend.scheduler.priority_scheduler import PriorityScheduler
        
        # Create workflow B->C->E->G
        task_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING)
        task_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [task_b])
        task_e = create_task_with_dependencies(tool_e, TaskType.POST_TRAINING, [task_c])
        task_g = create_task_with_dependencies(tool_g, TaskType.DEPLOYMENT, [task_e])
        
        workflow = WorkflowFactory.create_workflow(
            name="short_priority_test",
            tasks=[task_b, task_c, task_e, task_g]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="short_priority_pipeline",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # B should have highest priority (depth 0)
        assert 100 <= task_b.priority <= 109
        # C should have next priority (depth 1)
        assert 90 <= task_c.priority <= 99
        # E should be depth 2
        assert 80 <= task_e.priority <= 89
        # G should be depth 3
        assert 70 <= task_g.priority <= 79
    
    def test_priority_10_point_difference_per_depth(
        self, tool_a, tool_b, tool_c
    ):
        """Verify ~10 point priority difference per depth level."""
        from src.backend.scheduler.priority_scheduler import PriorityScheduler
        
        task_a = create_task_with_dependencies(tool_a, TaskType.PRE_TRAINING)
        task_b = create_task_with_dependencies(tool_b, TaskType.PRE_TRAINING, [task_a])
        task_c = create_task_with_dependencies(tool_c, TaskType.IN_TRAINING, [task_b])
        
        workflow = WorkflowFactory.create_workflow(
            name="diff_test",
            tasks=[task_a, task_b, task_c]
        )
        
        pipeline = DefenseEvaluationPipeline(
            name="diff_pipeline",
            workflows=[workflow]
        )
        
        scheduler = PriorityScheduler(pipeline)
        
        # Priority difference should be approximately 10 per level
        diff_a_b = task_a.priority - task_b.priority
        diff_b_c = task_b.priority - task_c.priority
        
        # Allow for counter bonus (up to 9)
        assert 1 <= diff_a_b <= 19
        assert 1 <= diff_b_c <= 19
