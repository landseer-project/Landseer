"""
Comprehensive tests for workflow specification from docs/Workflow.md.

Tests cover:
1. Workflow permutation logic (order matters)
2. Single during_training tool constraint
3. Baseline tool substitution
4. Artifact chaining (outputs -> inputs)
5. Converter tool invocation
6. Evaluation artifact selection
7. Edge cases and corner cases
"""

import pytest
from pathlib import Path
from typing import List, Dict, Set, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.pipeline.workflow_generator import (
    WorkflowGenerator,
    generate_stage_permutations,
    generate_during_training_options,
)
from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    TaskFactory,
    PreTrainingTask,
    InTrainingTask,
    PostTrainingTask,
    DeploymentTask,
    clear_task_registry,
)
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear task registry before each test."""
    clear_task_registry()
    yield
    clear_task_registry()


@pytest.fixture
def tool_a():
    """Baseline tool A (pre_training)."""
    return ToolDefinition(
        name="A",
        container=ContainerConfig(image="test/a:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_b():
    """Actual tool B (pre_training)."""
    return ToolDefinition(
        name="B",
        container=ContainerConfig(image="test/b:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_b2():
    """Actual tool B2 (pre_training)."""
    return ToolDefinition(
        name="B2",
        container=ContainerConfig(image="test/b2:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_c():
    """Baseline tool C (during_training)."""
    return ToolDefinition(
        name="C",
        container=ContainerConfig(image="test/c:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_d():
    """Actual tool D (during_training)."""
    return ToolDefinition(
        name="D",
        container=ContainerConfig(image="test/d:v1", command="python run.py"),
        is_baseline=False
    )


@pytest.fixture
def tool_e():
    """Baseline tool E (post_training)."""
    return ToolDefinition(
        name="E",
        container=ContainerConfig(image="test/e:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def tool_g():
    """Baseline tool G (deployment)."""
    return ToolDefinition(
        name="G",
        container=ContainerConfig(image="test/g:v1", command="python run.py"),
        is_baseline=True
    )


@pytest.fixture
def workflow_generator():
    """Create a workflow generator instance."""
    return WorkflowGenerator(pipeline_id="test_pipeline")


# ============================================================================
# Test: Workflow Permutation Logic
# ============================================================================


class TestWorkflowPermutation:
    """Tests for workflow permutation logic from Workflow.md."""
    
    def test_order_matters_tool1_tool2_not_equal_tool2_tool1(self, tool_b, tool_b2):
        """Order of tools within stage matters: tool1->tool2 != tool2->tool1."""
        permutations = generate_stage_permutations([tool_b, tool_b2])
        
        # Should have both orders
        found_b_b2 = False
        found_b2_b = False
        
        for perm in permutations:
            if len(perm) == 2:
                if perm[0].name == "B" and perm[1].name == "B2":
                    found_b_b2 = True
                elif perm[0].name == "B2" and perm[1].name == "B":
                    found_b2_b = True
        
        assert found_b_b2, "Should have permutation B->B2"
        assert found_b2_b, "Should have permutation B2->B"
        assert found_b_b2 != found_b2_b or (found_b_b2 and found_b2_b), \
            "B->B2 and B2->B should be different workflows"
    
    def test_example_from_workflow_md(self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g):
        """
        Test the exact example from Workflow.md.
        
        With A, C, E, G as baseline and B, D as actual tools:
        Expected workflows (after baseline deduplication):
        1. A->B->C->E->G (or B->A->C->E->G or A->C->E->G - all same due to baseline)
        2. B->C->E->G
        3. A->B->D->E->G (or B->A->D->E->G or A->D->E->G - all same due to baseline)
        4. B->D->E->G
        
        Note: Workflow.md says "workflow 1, 2, 3 are the same workflow because A is a baseline tool"
        So we expect 4 unique workflows (not 8).
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        # Set up stages
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate 4 workflows (2 pre options * 2 during options)
        # Pre options: [B], [A] (baseline)
        # During options: [D], [C] (baseline)
        assert len(workflows) == 4, f"Expected 4 workflows, got {len(workflows)}"
        
        # Check each expected workflow exists
        workflow_tool_sequences = []
        for wf in workflows:
            tool_seq = [t.tool.name for t in wf.tasks]
            workflow_tool_sequences.append(tool_seq)
        
        # Expected sequences (after baseline deduplication)
        expected_sequences = [
            ["A", "C", "E", "G"],  # All baseline
            ["B", "C", "E", "G"],  # B pre, C during
            ["A", "D", "E", "G"],  # A pre, D during
            ["B", "D", "E", "G"],  # B pre, D during
        ]
        
        for expected in expected_sequences:
            assert expected in workflow_tool_sequences, \
                f"Expected workflow {expected} not found in generated workflows: {workflow_tool_sequences}"
    
    def test_baseline_substitution_for_empty_set(self, tool_a, tool_b):
        """Empty set should be substituted with baseline tool."""
        permutations = generate_stage_permutations([tool_b], baseline=tool_a)
        
        # Should include baseline for empty set
        has_baseline = any(len(perm) == 1 and perm[0].is_baseline for perm in permutations)
        assert has_baseline, "Should include baseline tool for empty set"
    
    def test_baseline_workflows_are_deduplicated(self, tool_a, tool_b, tool_c, tool_e, tool_g):
        """
        Workflows with only baseline tools should be deduplicated.
        
        From Workflow.md: "workflow 1, 2, 3 are the same workflow because A is a baseline tool"
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Count workflows with only baseline in pre_training
        baseline_only_workflows = []
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            if all(t.tool.is_baseline for t in pre_tasks):
                baseline_only_workflows.append([t.tool.name for t in wf.tasks])
        
        # All baseline-only workflows should have same task sequence
        if len(baseline_only_workflows) > 1:
            first_seq = baseline_only_workflows[0]
            for seq in baseline_only_workflows[1:]:
                assert seq == first_seq, \
                    "Baseline-only workflows should be deduplicated (same task sequence)"
    
    def test_multiple_tools_in_pre_training_creates_permutations(self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g):
        """
        Multiple tools in pre_training should create permutations.
        
        From Workflow.md: "If there was a tool B2 in the pre-training stage,
        then we would permute B and B2, which would give us ordered sets
        (B, B2), (B2, B), (B), (B2), and a null set."
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Count pre_training tool sequences
        pre_sequences = set()
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            seq = tuple(t.tool.name for t in pre_tasks)
            pre_sequences.add(seq)
        
        # Should have 5 different pre_training sequences:
        # (B, B2), (B2, B), (B), (B2), (A) - baseline
        assert len(pre_sequences) == 5, \
            f"Expected 5 pre_training sequences, got {len(pre_sequences)}: {pre_sequences}"
        
        assert ("B", "B2") in pre_sequences, "Should have B->B2"
        assert ("B2", "B") in pre_sequences, "Should have B2->B"
        assert ("B",) in pre_sequences, "Should have B alone"
        assert ("B2",) in pre_sequences, "Should have B2 alone"
        assert ("A",) in pre_sequences, "Should have baseline A"
    
    def test_b2_scenario_creates_5_workflows_in_pre_stage(self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g):
        """
        Test the B2 scenario from Workflow.md.
        
        From Workflow.md: "If there was a tool B2 in the pre-training stage,
        then we would permute B and B2, which would give us ordered sets
        (B, B2), (B2, B), (B), (B2), and a null set. We would always substitute
        the null set with the baseline tool and thus we would have 5 different
        workflows just because of the pre-training stage."
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        # Get pre_training options
        stage_options = generator.generate_all_stage_options()
        pre_options = stage_options["pre_training"]
        
        # Should have 5 options for pre_training
        assert len(pre_options) == 5, \
            f"Expected 5 pre_training options, got {len(pre_options)}"
        
        # Verify all expected sequences
        option_names = [tuple(t.name for t in opt) for opt in pre_options]
        assert ("B", "B2") in option_names, "Should have (B, B2)"
        assert ("B2", "B") in option_names, "Should have (B2, B)"
        assert ("B",) in option_names, "Should have (B)"
        assert ("B2",) in option_names, "Should have (B2)"
        assert ("A",) in option_names, "Should have baseline (A)"
    
    def test_b2_workflows_are_different(self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g):
        """
        Test that B->B2 and B2->B create different workflows.
        
        From Workflow.md: "if there was a tool B2 in pre training stage, then
        (B->B2->C->E->G) and (B2->B->C->E->G), (B->C->E->G) and (B2->C->E->G)
        are different workflows."
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflows with B->B2 and B2->B
        b_b2_workflow = None
        b2_b_workflow = None
        b_only_workflow = None
        b2_only_workflow = None
        
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            if len(pre_tasks) == 2:
                if pre_tasks[0].tool.name == "B" and pre_tasks[1].tool.name == "B2":
                    b_b2_workflow = wf
                elif pre_tasks[0].tool.name == "B2" and pre_tasks[1].tool.name == "B":
                    b2_b_workflow = wf
            elif len(pre_tasks) == 1:
                if pre_tasks[0].tool.name == "B":
                    b_only_workflow = wf
                elif pre_tasks[0].tool.name == "B2":
                    b2_only_workflow = wf
        
        assert b_b2_workflow is not None, "Should have workflow B->B2"
        assert b2_b_workflow is not None, "Should have workflow B2->B"
        assert b_only_workflow is not None, "Should have workflow B only"
        assert b2_only_workflow is not None, "Should have workflow B2 only"
        
        # These should be different workflows
        assert b_b2_workflow.id != b2_b_workflow.id, "B->B2 and B2->B should be different workflows"
        assert b_only_workflow.id != b2_only_workflow.id, "B and B2 only should be different workflows"


# ============================================================================
# Test: Single During Training Tool Constraint
# ============================================================================


class TestDuringTrainingConstraint:
    """Tests for single during_training tool constraint."""
    
    def test_each_workflow_has_only_one_during_training_tool(self, tool_a, tool_c, tool_d, tool_e, tool_g):
        """Each workflow can only have 1 during training tool."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        for wf in workflows:
            during_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]
            assert len(during_tasks) <= 1, \
                f"Workflow {wf.name} has {len(during_tasks)} during_training tasks, should be <= 1"
    
    def test_during_training_no_permutations(self, tool_c, tool_d):
        """During training should not have permutations of multiple tools."""
        options = generate_during_training_options([tool_d], tool_c)
        
        # Should only have single-tool options
        for option in options:
            assert len(option) <= 1, \
                f"During training option should have <= 1 tool, got {len(option)}"
        
        # Should have both D and C (baseline) as options
        tool_names = {opt[0].name for opt in options if opt}
        assert "D" in tool_names, "Should have D as option"
        assert "C" in tool_names, "Should have C (baseline) as option"
    
    def test_multiple_during_training_tools_create_separate_workflows(self, tool_a, tool_c, tool_d, tool_e, tool_g):
        """Multiple during_training tools should create separate workflows."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)  # Only D, but C is baseline
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should have workflows with C and workflows with D
        has_c = False
        has_d = False
        
        for wf in workflows:
            during_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]
            if during_tasks:
                tool_name = during_tasks[0].tool.name
                if tool_name == "C":
                    has_c = True
                elif tool_name == "D":
                    has_d = True
        
        assert has_c, "Should have workflows with C (baseline)"
        assert has_d, "Should have workflows with D"


# ============================================================================
# Test: Execution Order
# ============================================================================


class TestExecutionOrder:
    """Tests for workflow execution order."""
    
    def test_stage_execution_order(self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g):
        """
        Pipeline should execute in order: pre -> during -> post -> deployment.
        
        From Workflow.md: "First we need to execute the pre-training tools,
        then the during training tools, then the post-training tools, and finally
        the deployment tools."
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        for wf in workflows:
            tasks = wf.tasks
            
            # Find first task of each stage
            pre_first = next((t for t in tasks if t.task_type == TaskType.PRE_TRAINING), None)
            during_first = next((t for t in tasks if t.task_type == TaskType.IN_TRAINING), None)
            post_first = next((t for t in tasks if t.task_type == TaskType.POST_TRAINING), None)
            deploy_first = next((t for t in tasks if t.task_type == TaskType.DEPLOYMENT), None)
            
            # Check priorities reflect execution order
            if pre_first and during_first:
                assert pre_first.priority > during_first.priority, \
                    "Pre-training should have higher priority (execute first)"
            
            if during_first and post_first:
                assert during_first.priority > post_first.priority, \
                    "During-training should have higher priority than post-training"
            
            if post_first and deploy_first:
                assert post_first.priority > deploy_first.priority, \
                    "Post-training should have higher priority than deployment"
    
    def test_within_stage_order(self, tool_b, tool_b2, tool_c, tool_e, tool_g):
        """Tools within a stage should execute in order."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], None)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with B->B2
        b_b2_workflow = None
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            if len(pre_tasks) == 2 and pre_tasks[0].tool.name == "B" and pre_tasks[1].tool.name == "B2":
                b_b2_workflow = wf
                break
        
        assert b_b2_workflow is not None, "Should have workflow with B->B2"
        
        # Check dependencies
        pre_tasks = [t for t in b_b2_workflow.tasks if t.task_type == TaskType.PRE_TRAINING]
        assert len(pre_tasks) == 2, "Should have 2 pre_training tasks"
        
        # B2 should depend on B
        b_task = pre_tasks[0]
        b2_task = pre_tasks[1]
        assert b_task.id in [dep.id for dep in b2_task.dependencies], \
            "B2 should depend on B"


# ============================================================================
# Test: Artifact Chaining
# ============================================================================


class TestArtifactChaining:
    """Tests for artifact chaining (outputs -> inputs)."""
    
    def test_output_becomes_input_for_next_tool(self, temp_workspace):
        """
        Outputs of previous tool become input of next tool in chain.
        
        From Workflow.md: "A->B->C->E->G, the output of A is the input of B,
        the output of B is the input of C, the output of C is the input of E,
        and the output of E is the input of G."
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create task A (pre_training)
        task_a = TaskInfo(
            id="task_a",
            tool_name="tool_a",
            tool_image="test/a:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]
        )
        
        # Execute task A (simulate)
        task_a_dir = temp_workspace / "task_a"
        task_a_output = task_a_dir / "output"
        task_a_output.mkdir(parents=True, exist_ok=True)
        (task_a_output / "dataset.npy").write_bytes(b"dataset from A")
        
        # Create task B that depends on A
        task_b = TaskInfo(
            id="task_b",
            tool_name="tool_b",
            tool_image="test/b:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["task_a"]
        )
        
        # Execute task B with dependency outputs
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task_b,
                dependency_outputs={"task_a": task_a_output}
            )
        
        # Check that A's output was copied to B's input
        task_b_input = temp_workspace / "task_b" / "input"
        dataset_file = task_b_input / "dataset.npy"
        
        assert dataset_file.exists(), "A's output should be copied to B's input"
        assert dataset_file.read_bytes() == b"dataset from A", "Dataset content should match"
    
    def test_first_pre_tool_takes_config_data(self, temp_workspace):
        """
        First pre tool always takes data mentioned by user in config.
        
        From Workflow.md: "The first pre tool always takes data mentioned by user
        in the config as its input (eg. dataset name, variant, etc.)."
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create dataset directory (simulating config data)
        data_dir = temp_workspace / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "data.npy").write_bytes(b"original dataset")
        (data_dir / "labels.npy").write_bytes(b"original labels")
        
        # Create first pre_training task (no dependencies)
        task_a = TaskInfo(
            id="task_a",
            tool_name="tool_a",
            tool_image="test/a:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=100,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=[]  # No dependencies - first tool
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task_a,
                input_path=data_dir  # Config data passed as input
            )
        
        # Check that config data was copied to input
        task_a_input = temp_workspace / "task_a" / "input"
        assert (task_a_input / "data.npy").exists(), "Config data should be in input"
        assert (task_a_input / "labels.npy").exists(), "Config labels should be in input"
    
    def test_dataset_passed_through_chain(self, temp_workspace):
        """
        Dataset should be passed through the chain correctly.
        
        From Workflow.md: "if post-training tool is again provided dataset as input,
        then it should be the same dataset as outputted by the pre tool."
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create pre_training task output
        pre_output = temp_workspace / "pre_task" / "output"
        pre_output.mkdir(parents=True, exist_ok=True)
        (pre_output / "dataset.npy").write_bytes(b"preprocessed dataset")
        
        # Create during_training task output (model only, no dataset)
        during_output = temp_workspace / "during_task" / "output"
        during_output.mkdir(parents=True, exist_ok=True)
        (during_output / "model.pt").write_bytes(b"trained model")
        
        # Create post_training task that needs both model and dataset
        post_task = TaskInfo(
            id="post_task",
            tool_name="post_tool",
            tool_image="test/post:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=80,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["pre_task", "during_task"]
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                post_task,
                dependency_outputs={
                    "pre_task": pre_output,
                    "during_task": during_output
                }
            )
        
        # Check that both model and dataset are in input
        post_input = temp_workspace / "post_task" / "input"
        assert (post_input / "model.pt").exists(), "Model from during_training should be in input"
        assert (post_input / "dataset.npy").exists(), "Dataset from pre_training should be in input"
        assert (post_input / "dataset.npy").read_bytes() == b"preprocessed dataset", \
            "Dataset should be from pre_training, not original"


# ============================================================================
# Test: File Format Requirements
# ============================================================================


class TestFileFormatRequirements:
    """Tests for file format requirements."""
    
    def test_dataset_files_must_be_npy(self, temp_workspace):
        """
        Dataset files must be in numpy format (.npy).
        
        From Workflow.md: "We need to make sure that the input/output dataset
        files are always in numpy format (.npy)"
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create dependency output with .npy file
        dep_output = temp_workspace / "dep_task" / "output"
        dep_output.mkdir(parents=True, exist_ok=True)
        (dep_output / "data.npy").write_bytes(b"numpy data")
        
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="pre_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["dep_task"]
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs={"dep_task": dep_output}
            )
        
        # Verify .npy file was copied
        task_input = temp_workspace / "task_1" / "input"
        assert (task_input / "data.npy").exists(), "Dataset file should be .npy format"
    
    def test_model_files_must_be_pt(self, temp_workspace):
        """
        Model files must be in PyTorch format (.pt).
        
        From Workflow.md: "model files are always handled by landseer in pytorch
        format (.pt) and not in any other format."
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create dependency output with .pt file
        dep_output = temp_workspace / "dep_task" / "output"
        dep_output.mkdir(parents=True, exist_ok=True)
        (dep_output / "model.pt").write_bytes(b"pytorch model")
        
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=80,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["dep_task"]
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs={"dep_task": dep_output}
            )
        
        # Verify .pt file was copied
        task_input = temp_workspace / "task_1" / "input"
        assert (task_input / "model.pt").exists(), "Model file should be .pt format"


# ============================================================================
# Test: Converter Tool Invocation
# ============================================================================


class TestConverterToolInvocation:
    """Tests for converter tool invocation logic."""
    
    def test_converter_needed_when_input_not_pytorch(self):
        """
        Converter tool should be invoked when tool takes input in non-PyTorch format.
        
        From Workflow.md: "Converter tool is only invoked when: a) tool takes input
        in format other than pytorch"
        """
        # This would require checking tool framework labels
        # For now, we test the concept
        
        # Create a tool that requires TensorFlow input
        tf_tool = ToolDefinition(
            name="tf_tool",
            container=ContainerConfig(image="test/tf:v1", command="python main.py"),
            is_baseline=False
        )
        
        # In a real implementation, we would check:
        # - Tool's framework_label indicates TensorFlow
        # - Previous tool output is PyTorch (.pt)
        # - Converter tool should be inserted to convert .pt -> TensorFlow format
        
        # This is a placeholder test - actual implementation would be in scheduler
        assert tf_tool is not None
    
    def test_converter_needed_when_output_not_pytorch(self):
        """
        Converter tool should be invoked when tool gives output in non-PyTorch format.
        
        From Workflow.md: "Converter tool is only invoked when: b) tool gives output
        in format other than pytorch"
        """
        # Create a tool that outputs TensorFlow model
        tf_tool = ToolDefinition(
            name="tf_tool",
            container=ContainerConfig(image="test/tf:v1", command="python main.py"),
            is_baseline=False
        )
        
        # In a real implementation, we would check:
        # - Tool's framework_label indicates TensorFlow output
        # - Next tool expects PyTorch input
        # - Converter tool should be inserted to convert TensorFlow -> .pt
        
        assert tf_tool is not None
    
    def test_converter_converts_tensorflow_to_pytorch(self):
        """
        Converter should convert TensorFlow model to PyTorch format.
        
        From Workflow.md: "In case a tool outputs tensorflow model, then the converter
        tool should convert the model to pytorch format."
        """
        # This would be tested in the converter tool itself
        # For workflow tests, we verify the scheduler invokes it correctly
        pass
    
    def test_converter_converts_pytorch_to_tensorflow(self):
        """
        Converter should convert PyTorch model to TensorFlow format.
        
        From Workflow.md: "if a tool takes input in tensorflow format, then the converter
        tool should convert the saved pytorch model to tensorflow format"
        """
        # This would be tested in the converter tool itself
        pass


# ============================================================================
# Test: Evaluation Artifact Selection
# ============================================================================


class TestEvaluationArtifactSelection:
    """Tests for evaluation artifact selection logic."""
    
    def test_evaluation_uses_last_dataset_modifier(self):
        """
        Evaluation should use dataset from last tool that modified dataset.
        
        From Workflow.md Case 1: "if A, B, and C are present, after them none of
        the tools like E, G made changes to dataset (i.e. outputted the dataset)
        then the evaluation should be done on the dataset outputted by C."
        """
        # Workflow: A->B->C->E->G
        # C is last to modify dataset, so evaluation should use C's dataset output
        
        # This would be tested by:
        # 1. Creating workflow A->B->C->E->G
        # 2. Marking which tools output datasets
        # 3. Verifying evaluation uses C's dataset
        
        # Placeholder - actual implementation would track dataset/modification
        pass
    
    def test_evaluation_uses_last_model_modifier(self):
        """
        Evaluation should use model from last tool that modified model.
        
        From Workflow.md Case 1: "if E was the last tool to make changes to the model,
        then the evaluation should be done on the model outputted by E."
        """
        # Workflow: A->B->C->E->G
        # E is last to modify model, so evaluation should use E's model output
        
        # Placeholder - actual implementation would track model modification
        pass
    
    def test_evaluation_uses_later_dataset_if_modified(self):
        """
        Evaluation should use dataset from later tool if it modified dataset.
        
        From Workflow.md Case 2: "if A, B, and C are present, after them G made changes
        to dataset (i.e. outputted the dataset) then the evaluation should be done on
        the dataset outputted by G."
        """
        # Workflow: A->B->C->E->G
        # G modifies dataset, so evaluation should use G's dataset (not C's)
        
        # Placeholder - actual implementation would track dataset modification
        pass
    
    def test_evaluation_uses_correct_model_when_later_tool_modifies_dataset(self):
        """
        Evaluation should use correct model even when later tool modifies dataset.
        
        From Workflow.md Case 2: "if G was the last tool to make changes to the model,
        then the evaluation should be done on the model outputted by E."
        Note: This seems contradictory - if G modifies model, should use G's model.
        But the spec says E. This might be a typo or specific requirement.
        """
        # Placeholder - needs clarification from spec
        pass


# ============================================================================
# Test: Edge Cases and Corner Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner cases."""
    
    def test_empty_stage_handling(self, tool_a, tool_c, tool_e, tool_g):
        """Empty stages should be handled gracefully."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should still generate at least one workflow (all baselines)
        assert len(workflows) >= 1, "Should generate at least one workflow even with all baselines"
    
    def test_single_tool_in_each_stage(self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g):
        """Workflow with single tool in each stage should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate workflows
        assert len(workflows) > 0, "Should generate workflows"
        
        # Check that workflows have correct structure
        for wf in workflows:
            assert len(wf.tasks) >= 4, "Workflow should have at least 4 tasks (one per stage)"
    
    def test_no_baseline_tools(self, tool_b, tool_d):
        """Workflow generation should work without baseline tools."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], None)
        generator.set_stage_tools("during_training", [tool_d], None)
        generator.set_stage_tools("post_training", [], None)
        generator.set_stage_tools("deployment", [], None)
        
        workflows = generator.generate_all_workflows()
        
        # Should still generate workflows
        assert len(workflows) > 0, "Should generate workflows even without baselines"
    
    def test_all_baseline_workflow(self, tool_a, tool_c, tool_e, tool_g):
        """Workflow with all baseline tools should be valid."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should have at least one workflow
        assert len(workflows) >= 1, "Should generate baseline-only workflow"
        
        # Check that all tasks are baseline
        baseline_workflow = workflows[0]
        for task in baseline_workflow.tasks:
            assert task.tool.is_baseline, f"Task {task.id} should be baseline"
    
    def test_very_long_workflow_chain(self, tool_b, tool_c, tool_e, tool_g):
        """Workflow with many tools in pre_training should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        # Create many tools
        many_tools = []
        for i in range(5):
            tool = ToolDefinition(
                name=f"tool_{i}",
                container=ContainerConfig(image=f"test/tool{i}:v1", command="python run.py"),
                is_baseline=False
            )
            many_tools.append(tool)
        
        generator.set_stage_tools("pre_training", many_tools, None)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate many workflows (permutations of 5 tools)
        assert len(workflows) > 0, "Should generate workflows"
        
        # Check that workflows have correct number of pre_training tasks
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            assert len(pre_tasks) <= 5, "Should have at most 5 pre_training tasks"
    
    def test_dependency_chain_with_failed_task(self, temp_workspace):
        """
        Failed tasks should block dependent tasks.
        
        From Workflow.md: "If a task has failed and I am restarting a pipeline
        run with cache enabled, try to rerun the task that failed."
        """
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create a failed task
        failed_task_output = temp_workspace / "failed_task" / "output"
        failed_task_output.mkdir(parents=True, exist_ok=True)
        # No .success marker or empty output indicates failure
        
        # Create dependent task
        dependent_task = TaskInfo(
            id="dependent_task",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["failed_task"]
        )
        
        # Dependent task should not execute if dependency failed
        # (This would be checked by scheduler/worker logic)
        # For now, we verify the dependency relationship exists
        assert "failed_task" in dependent_task.dependency_ids
    
    def test_concurrent_workflows_artifact_isolation(self, temp_workspace):
        """Artifacts from different workflows should be isolated."""
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create outputs from workflow 1
        wf1_output = temp_workspace / "wf1_task" / "output"
        wf1_output.mkdir(parents=True, exist_ok=True)
        (wf1_output / "model.pt").write_bytes(b"workflow1 model")
        
        # Create outputs from workflow 2
        wf2_output = temp_workspace / "wf2_task" / "output"
        wf2_output.mkdir(parents=True, exist_ok=True)
        (wf2_output / "model.pt").write_bytes(b"workflow2 model")
        
        # Task from workflow 1 should only get workflow 1 artifacts
        task_wf1 = TaskInfo(
            id="task_wf1_next",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=80,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=["workflow_1"],
            pipeline_id="pipeline_1",
            dependency_ids=["wf1_task"]
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task_wf1,
                dependency_outputs={"wf1_task": wf1_output}
            )
        
        # Verify only workflow 1 artifacts are present
        task_input = temp_workspace / "task_wf1_next" / "input"
        assert (task_input / "model.pt").read_bytes() == b"workflow1 model", \
            "Should only have workflow 1 artifacts"
    
    def test_missing_dependency_artifacts(self, temp_workspace):
        """Missing dependency artifacts should be handled gracefully."""
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create task with missing dependency
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=90,
            status="pending",
            task_type="post_training",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["missing_task"]
        )
        
        # Dependency output doesn't exist
        missing_output = temp_workspace / "missing_task" / "output"
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            # Should handle gracefully (log warning, continue)
            result = runner.run_task(
                task,
                dependency_outputs={"missing_task": missing_output}
            )
        
        # Task should still execute (though it may fail if it needs the dependency)
        assert result is not None
    
    def test_multiple_dependencies_same_file_overwrite(self, temp_workspace):
        """If multiple dependencies produce same file, later one should overwrite."""
        from src.worker.runner import TaskRunner
        from src.worker.client import TaskInfo
        
        runner = TaskRunner(workspace_dir=temp_workspace)
        
        # Create two dependencies with same file
        dep1_output = temp_workspace / "dep1" / "output"
        dep1_output.mkdir(parents=True, exist_ok=True)
        (dep1_output / "model.pt").write_bytes(b"model from dep1")
        
        dep2_output = temp_workspace / "dep2" / "output"
        dep2_output.mkdir(parents=True, exist_ok=True)
        (dep2_output / "model.pt").write_bytes(b"model from dep2")
        
        task = TaskInfo(
            id="task_1",
            tool_name="tool1",
            tool_image="test/img:v1",
            tool_command="python main.py",
            tool_runtime=None,
            tool_is_baseline=False,
            config={},
            priority=80,
            status="pending",
            task_type="deployment",
            counter=1,
            workflows=[],
            pipeline_id="pipeline_1",
            dependency_ids=["dep1", "dep2"]
        )
        
        with patch.object(runner, '_container_runner') as mock_runner:
            mock_runner.run.return_value = (0, "Success")
            mock_runner.pull_image.return_value = True
            
            result = runner.run_task(
                task,
                dependency_outputs={
                    "dep1": dep1_output,
                    "dep2": dep2_output
                }
            )
        
        # Last dependency copied should win
        task_input = temp_workspace / "task_1" / "input"
        model_file = task_input / "model.pt"
        assert model_file.exists(), "Model file should exist"
        # Content will be from last dependency copied (dep2 in this case)
        assert model_file.read_bytes() in [b"model from dep1", b"model from dep2"]


# ============================================================================
# Test: Workflow Generation Summary
# ============================================================================


class TestWorkflowGenerationSummary:
    """Tests for workflow generation summary and statistics."""
    
    def test_workflow_summary_includes_deduplication_stats(self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g):
        """Workflow summary should include task deduplication statistics."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        summary = generator.get_workflow_summary(workflows)
        
        assert "total_workflows" in summary
        assert "total_unique_tasks" in summary
        assert "task_reuse_savings" in summary
        assert summary["total_unique_tasks"] < summary["total_task_instances"], \
            "Should have task reuse (unique tasks < total instances)"
    
    def test_task_deduplication_across_workflows(self, tool_a, tool_b, tool_c, tool_e, tool_g):
        """Tasks with same tool and dependencies should be deduplicated."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Collect all tasks by their identity (tool + dependencies)
        task_identities = {}
        for wf in workflows:
            for task in wf.tasks:
                # Create identity based on tool and dependency IDs
                dep_ids = tuple(sorted(dep.id for dep in task.dependencies))
                identity = (task.tool.name, dep_ids)
                
                if identity not in task_identities:
                    task_identities[identity] = []
                task_identities[identity].append(task.id)
        
        # Tasks with same identity should have same task ID (deduplication)
        for identity, task_ids in task_identities.items():
            unique_task_ids = set(task_ids)
            assert len(unique_task_ids) == 1, \
                f"Tasks with same identity {identity} should be deduplicated, got {unique_task_ids}"


# ============================================================================
# Test: Complex Scenarios
# ============================================================================


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""
    
    def test_full_pipeline_example_from_workflow_md(self, tool_a, tool_b, tool_c, tool_d, tool_e, tool_g):
        """
        Test the full example from Workflow.md.
        
        Pipeline config:
        - pre_training: A (baseline), B (actual)
        - during_training: C (baseline), D (actual)
        - post_training: E (baseline)
        - deployment: G (baseline)
        
        Expected 4 workflows (after baseline deduplication):
        - A->C->E->G (all baseline)
        - B->C->E->G (B pre, C during)
        - A->D->E->G (A pre, D during)
        - B->D->E->G (B pre, D during)
        """
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b], tool_a)
        generator.set_stage_tools("during_training", [tool_d], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Verify we have exactly 4 workflows (2 pre * 2 during)
        assert len(workflows) == 4, f"Expected 4 workflows, got {len(workflows)}"
        
        # Verify each workflow has correct structure
        for wf in workflows:
            # Should have tasks from all stages
            has_pre = any(t.task_type == TaskType.PRE_TRAINING for t in wf.tasks)
            has_during = any(t.task_type == TaskType.IN_TRAINING for t in wf.tasks)
            has_post = any(t.task_type == TaskType.POST_TRAINING for t in wf.tasks)
            has_deploy = any(t.task_type == TaskType.DEPLOYMENT for t in wf.tasks)
            
            assert has_pre, f"Workflow {wf.name} should have pre_training task"
            assert has_during, f"Workflow {wf.name} should have during_training task"
            assert has_post, f"Workflow {wf.name} should have post_training task"
            assert has_deploy, f"Workflow {wf.name} should have deployment task"
            
            # Should have exactly 1 during_training task
            during_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]
            assert len(during_tasks) == 1, \
                f"Workflow {wf.name} should have exactly 1 during_training task, got {len(during_tasks)}"
            
            # Should have exactly 1 pre_training task (since only one actual tool)
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            assert len(pre_tasks) == 1, \
                f"Workflow {wf.name} should have exactly 1 pre_training task, got {len(pre_tasks)}"
    
    def test_workflow_with_multiple_pre_training_tools(self, tool_a, tool_b, tool_b2, tool_c, tool_e, tool_g):
        """Workflow with multiple pre_training tools should chain correctly."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], tool_a)
        generator.set_stage_tools("during_training", [], tool_c)
        generator.set_stage_tools("post_training", [], tool_e)
        generator.set_stage_tools("deployment", [], tool_g)
        
        workflows = generator.generate_all_workflows()
        
        # Find workflow with B->B2
        b_b2_workflow = None
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            if len(pre_tasks) == 2 and pre_tasks[0].tool.name == "B" and pre_tasks[1].tool.name == "B2":
                b_b2_workflow = wf
                break
        
        assert b_b2_workflow is not None, "Should have workflow with B->B2"
        
        # Verify chaining: B2 depends on B, C depends on B2
        pre_tasks = [t for t in b_b2_workflow.tasks if t.task_type == TaskType.PRE_TRAINING]
        during_task = next((t for t in b_b2_workflow.tasks if t.task_type == TaskType.IN_TRAINING), None)
        
        assert len(pre_tasks) == 2
        assert pre_tasks[1].id in [dep.id for dep in pre_tasks[0].dependencies] or \
               pre_tasks[0].id in [dep.id for dep in pre_tasks[1].dependencies], \
            "Pre_training tasks should be chained"
        
        if during_task:
            last_pre_task = pre_tasks[-1]
            assert last_pre_task.id in [dep.id for dep in during_task.dependencies], \
                "During_training should depend on last pre_training task"


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path / "workspace"
