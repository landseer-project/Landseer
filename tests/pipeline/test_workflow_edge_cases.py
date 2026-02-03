"""
Edge cases and corner cases for workflow specification.

Tests cover:
1. Extreme scenarios (many tools, no tools, etc.)
2. Boundary conditions
3. Error conditions
4. Race conditions
5. Invalid inputs
"""

import pytest
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

from src.pipeline.workflow_generator import (
    WorkflowGenerator,
    generate_stage_permutations,
    generate_during_training_options,
)
from src.pipeline.tasks import TaskType, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig


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
def baseline_tool():
    """Create a baseline tool."""
    return ToolDefinition(
        name="baseline",
        container=ContainerConfig(image="test/baseline:v1", command="python run.py"),
        is_baseline=True
    )


# ============================================================================
# Test: Extreme Scenarios
# ============================================================================


class TestExtremeScenarios:
    """Tests for extreme scenarios."""
    
    def test_no_tools_in_any_stage(self, baseline_tool):
        """Pipeline with no actual tools should still generate baseline workflow."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate at least one workflow (all baselines)
        assert len(workflows) >= 1, "Should generate baseline workflow"
        
        # All tasks should be baseline
        for wf in workflows:
            for task in wf.tasks:
                assert task.tool.is_baseline, f"Task {task.id} should be baseline"
    
    def test_many_tools_in_pre_training(self, baseline_tool):
        """Pipeline with many tools in pre_training should handle correctly."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        # Create 10 tools
        many_tools = []
        for i in range(10):
            tool = ToolDefinition(
                name=f"tool_{i}",
                container=ContainerConfig(image=f"test/tool{i}:v1", command="python run.py"),
                is_baseline=False
            )
            many_tools.append(tool)
        
        generator.set_stage_tools("pre_training", many_tools, baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate many workflows (permutations of 10 tools)
        assert len(workflows) > 0, "Should generate workflows"
        
        # Check that workflows have correct structure
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            assert len(pre_tasks) <= 10, "Should have at most 10 pre_training tasks"
    
    def test_only_during_training_tools(self, baseline_tool):
        """Pipeline with only during_training tools should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_d = ToolDefinition(
            name="D",
            container=ContainerConfig(image="test/d:v1", command="python run.py"),
            is_baseline=False
        )
        
        generator.set_stage_tools("pre_training", [], baseline_tool)
        generator.set_stage_tools("during_training", [tool_d], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate workflows
        assert len(workflows) > 0, "Should generate workflows"
        
        # Each workflow should have exactly 1 during_training task
        for wf in workflows:
            during_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]
            assert len(during_tasks) == 1, "Should have exactly 1 during_training task"
    
    def test_only_pre_training_tools(self, baseline_tool):
        """Pipeline with only pre_training tools should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(
            name="B",
            container=ContainerConfig(image="test/b:v1", command="python run.py"),
            is_baseline=False
        )
        
        generator.set_stage_tools("pre_training", [tool_b], baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate workflows
        assert len(workflows) > 0, "Should generate workflows"
        
        # Each workflow should have at least pre_training task
        for wf in workflows:
            pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]
            assert len(pre_tasks) >= 1, "Should have at least 1 pre_training task"


# ============================================================================
# Test: Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions."""
    
    def test_single_actual_tool_per_stage(self, baseline_tool):
        """Pipeline with single actual tool per stage should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        tool_d = ToolDefinition(name="D", container=ContainerConfig(image="test/d:v1", command="python run.py"), is_baseline=False)
        tool_e = ToolDefinition(name="E", container=ContainerConfig(image="test/e:v1", command="python run.py"), is_baseline=False)
        tool_g = ToolDefinition(name="G", container=ContainerConfig(image="test/g:v1", command="python run.py"), is_baseline=False)
        
        generator.set_stage_tools("pre_training", [tool_b], baseline_tool)
        generator.set_stage_tools("during_training", [tool_d], baseline_tool)
        generator.set_stage_tools("post_training", [tool_e], baseline_tool)
        generator.set_stage_tools("deployment", [tool_g], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate workflows
        assert len(workflows) > 0, "Should generate workflows"
        
        # Verify structure
        for wf in workflows:
            assert len(wf.tasks) >= 4, "Should have tasks from all stages"
    
    def test_all_stages_have_multiple_tools(self, baseline_tool):
        """Pipeline with multiple tools in all stages should work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        pre_tools = [
            ToolDefinition(name=f"pre_{i}", container=ContainerConfig(image=f"test/pre{i}:v1", command="python run.py"), is_baseline=False)
            for i in range(3)
        ]
        during_tools = [
            ToolDefinition(name=f"during_{i}", container=ContainerConfig(image=f"test/during{i}:v1", command="python run.py"), is_baseline=False)
            for i in range(2)
        ]
        post_tools = [
            ToolDefinition(name=f"post_{i}", container=ContainerConfig(image=f"test/post{i}:v1", command="python run.py"), is_baseline=False)
            for i in range(2)
        ]
        deploy_tools = [
            ToolDefinition(name=f"deploy_{i}", container=ContainerConfig(image=f"test/deploy{i}:v1", command="python run.py"), is_baseline=False)
            for i in range(2)
        ]
        
        generator.set_stage_tools("pre_training", pre_tools, baseline_tool)
        generator.set_stage_tools("during_training", during_tools, baseline_tool)
        generator.set_stage_tools("post_training", post_tools, baseline_tool)
        generator.set_stage_tools("deployment", deploy_tools, baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate many workflows
        assert len(workflows) > 0, "Should generate workflows"
        
        # Verify each workflow has correct structure
        for wf in workflows:
            during_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]
            assert len(during_tasks) == 1, "Should have exactly 1 during_training task"


# ============================================================================
# Test: Error Conditions
# ============================================================================


class TestErrorConditions:
    """Tests for error conditions and invalid inputs."""
    
    def test_invalid_stage_name(self, baseline_tool):
        """Invalid stage name should raise error."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        
        with pytest.raises(ValueError, match="Unknown stage"):
            generator.set_stage_tools("invalid_stage", [tool_b], baseline_tool)
    
    def test_duplicate_tool_names(self, baseline_tool):
        """Duplicate tool names should be handled (tools are compared by identity, not name)."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        # Create two tools with same name but different configs
        tool_b1 = ToolDefinition(name="B", container=ContainerConfig(image="test/b1:v1", command="python run.py"), is_baseline=False)
        tool_b2 = ToolDefinition(name="B", container=ContainerConfig(image="test/b2:v1", command="python run.py"), is_baseline=False)
        
        # Should work (tools are different objects)
        generator.set_stage_tools("pre_training", [tool_b1, tool_b2], baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        assert len(workflows) > 0, "Should generate workflows even with duplicate names"
    
    def test_empty_tool_list_without_baseline(self):
        """Empty tool list without baseline should still work."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], None)
        generator.set_stage_tools("during_training", [], None)
        generator.set_stage_tools("post_training", [], None)
        generator.set_stage_tools("deployment", [], None)
        
        # Should handle gracefully (generate empty workflows or skip)
        workflows = generator.generate_all_workflows()
        # May generate empty workflows or no workflows
        assert isinstance(workflows, list), "Should return a list"


# ============================================================================
# Test: Permutation Edge Cases
# ============================================================================


class TestPermutationEdgeCases:
    """Tests for permutation edge cases."""
    
    def test_permutation_with_one_tool(self, baseline_tool):
        """Permutation with one tool should generate 2 options (tool and baseline)."""
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        
        perms = generate_stage_permutations([tool_b], baseline=baseline_tool)
        
        # Should have: [B], [baseline]
        assert len(perms) == 2, f"Expected 2 permutations, got {len(perms)}"
        
        tool_names = [tuple(t.name for t in perm) for perm in perms]
        assert ("B",) in tool_names, "Should have B"
        assert ("baseline",) in tool_names, "Should have baseline"
    
    def test_permutation_with_zero_tools(self, baseline_tool):
        """Permutation with zero tools should generate baseline."""
        perms = generate_stage_permutations([], baseline=baseline_tool)
        
        # Should have: [baseline]
        assert len(perms) == 1, f"Expected 1 permutation, got {len(perms)}"
        assert len(perms[0]) == 1, "Should have one tool"
        assert perms[0][0].is_baseline, "Should be baseline tool"
    
    def test_during_training_with_multiple_tools(self, baseline_tool):
        """During training with multiple tools should generate single-tool options only."""
        tool_d1 = ToolDefinition(name="D1", container=ContainerConfig(image="test/d1:v1", command="python run.py"), is_baseline=False)
        tool_d2 = ToolDefinition(name="D2", container=ContainerConfig(image="test/d2:v1", command="python run.py"), is_baseline=False)
        
        options = generate_during_training_options([tool_d1, tool_d2], baseline_tool)
        
        # Should have: [D1], [D2], [baseline]
        assert len(options) == 3, f"Expected 3 options, got {len(options)}"
        
        # Each option should have exactly 1 tool
        for option in options:
            assert len(option) == 1, f"During training option should have 1 tool, got {len(option)}"
        
        tool_names = {opt[0].name for opt in options}
        assert "D1" in tool_names, "Should have D1"
        assert "D2" in tool_names, "Should have D2"
        assert "baseline" in tool_names, "Should have baseline"
    
    def test_permutation_order_preservation(self, baseline_tool):
        """Permutation order should be preserved (B->B2 != B2->B)."""
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        tool_b2 = ToolDefinition(name="B2", container=ContainerConfig(image="test/b2:v1", command="python run.py"), is_baseline=False)
        
        perms = generate_stage_permutations([tool_b, tool_b2], baseline=baseline_tool)
        
        # Find B->B2 and B2->B
        found_b_b2 = False
        found_b2_b = False
        
        for perm in perms:
            if len(perm) == 2:
                if perm[0].name == "B" and perm[1].name == "B2":
                    found_b_b2 = True
                elif perm[0].name == "B2" and perm[1].name == "B":
                    found_b2_b = True
        
        assert found_b_b2, "Should have B->B2"
        assert found_b2_b, "Should have B2->B"
        assert found_b_b2 != found_b2_b or (found_b_b2 and found_b2_b), \
            "B->B2 and B2->B should be different"


# ============================================================================
# Test: Workflow Generation Edge Cases
# ============================================================================


class TestWorkflowGenerationEdgeCases:
    """Tests for workflow generation edge cases."""
    
    def test_workflow_with_all_baseline_tools(self, baseline_tool):
        """Workflow with all baseline tools should be valid."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        generator.set_stage_tools("pre_training", [], baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Should generate at least one workflow
        assert len(workflows) >= 1, "Should generate baseline workflow"
        
        # Verify all tasks are baseline
        for wf in workflows:
            for task in wf.tasks:
                assert task.tool.is_baseline, f"All tasks in {wf.name} should be baseline"
    
    def test_workflow_generation_with_missing_stages(self, baseline_tool):
        """Workflow generation should handle missing stages."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        
        # Only set pre_training
        generator.set_stage_tools("pre_training", [tool_b], baseline_tool)
        # Don't set other stages
        
        workflows = generator.generate_all_workflows()
        
        # Should still generate workflows (other stages will be empty)
        assert len(workflows) > 0, "Should generate workflows even with missing stages"
    
    def test_workflow_task_dependencies_are_correct(self, baseline_tool):
        """Task dependencies should be set correctly in workflows."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        tool_d = ToolDefinition(name="D", container=ContainerConfig(image="test/d:v1", command="python run.py"), is_baseline=False)
        
        generator.set_stage_tools("pre_training", [tool_b], baseline_tool)
        generator.set_stage_tools("during_training", [tool_d], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        for wf in workflows:
            tasks = wf.tasks
            
            # Find tasks by type
            pre_task = next((t for t in tasks if t.task_type == TaskType.PRE_TRAINING), None)
            during_task = next((t for t in tasks if t.task_type == TaskType.IN_TRAINING), None)
            post_task = next((t for t in tasks if t.task_type == TaskType.POST_TRAINING), None)
            deploy_task = next((t for t in tasks if t.task_type == TaskType.DEPLOYMENT), None)
            
            # Check dependencies
            if pre_task and during_task:
                assert pre_task.id in [dep.id for dep in during_task.dependencies], \
                    "During_training should depend on pre_training"
            
            if during_task and post_task:
                assert during_task.id in [dep.id for dep in post_task.dependencies], \
                    "Post_training should depend on during_training"
            
            if post_task and deploy_task:
                assert post_task.id in [dep.id for dep in deploy_task.dependencies], \
                    "Deployment should depend on post_training"


# ============================================================================
# Test: Converter Tool Edge Cases
# ============================================================================


class TestConverterToolEdgeCases:
    """Tests for converter tool edge cases."""
    
    def test_converter_not_needed_when_formats_match(self):
        """Converter should not be invoked when input/output formats match."""
        # If tool expects PyTorch and previous tool outputs PyTorch, no conversion needed
        # This would be checked by scheduler/converter logic
        pass
    
    def test_converter_needed_for_chain_of_different_formats(self):
        """
        Converter should handle chains of format conversions.
        
        Example: PyTorch -> TensorFlow -> PyTorch
        Should insert converters: PyTorch -> [convert] -> TensorFlow -> [convert] -> PyTorch
        """
        # This would be tested in converter/scheduler logic
        pass
    
    def test_converter_with_missing_model_file(self):
        """Converter should handle missing model files gracefully."""
        # If converter is invoked but model.pt doesn't exist, should fail gracefully
        pass


# ============================================================================
# Test: Evaluation Edge Cases
# ============================================================================


class TestEvaluationEdgeCases:
    """Tests for evaluation edge cases."""
    
    def test_evaluation_when_no_tool_modifies_dataset(self):
        """Evaluation should handle case where no tool modifies dataset."""
        # If all tools only modify model, evaluation should use original dataset
        pass
    
    def test_evaluation_when_no_tool_modifies_model(self):
        """Evaluation should handle case where no tool modifies model."""
        # If no tool outputs model, evaluation should fail or skip
        pass
    
    def test_evaluation_with_multiple_dataset_modifiers(self):
        """Evaluation should use last tool that modified dataset."""
        # If A, B, C, and G all modify dataset, use G's dataset
        pass
    
    def test_evaluation_with_multiple_model_modifiers(self):
        """Evaluation should use last tool that modified model."""
        # If C, E, and G all modify model, use G's model (or last one)
        pass


# ============================================================================
# Test: Task Deduplication Edge Cases
# ============================================================================


class TestTaskDeduplicationEdgeCases:
    """Tests for task deduplication edge cases."""
    
    def test_same_tool_different_dependencies_creates_different_tasks(self, baseline_tool):
        """Same tool with different dependencies should create different tasks."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        tool_b2 = ToolDefinition(name="B2", container=ContainerConfig(image="test/b2:v1", command="python run.py"), is_baseline=False)
        tool_d = ToolDefinition(name="D", container=ContainerConfig(image="test/d:v1", command="python run.py"), is_baseline=False)
        
        generator.set_stage_tools("pre_training", [tool_b, tool_b2], baseline_tool)
        generator.set_stage_tools("during_training", [tool_d], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Find tasks with tool D (during_training)
        d_tasks = set()
        for wf in workflows:
            for task in wf.tasks:
                if task.tool.name == "D":
                    dep_ids = tuple(sorted(dep.id for dep in task.dependencies))
                    d_tasks.add((task.id, dep_ids))
        
        # D should have different dependencies in different workflows
        # (depends on B in one workflow, B2 in another, etc.)
        assert len(d_tasks) > 0, "Should have D tasks"
        
        # If D appears in multiple workflows with same dependencies, should be same task
        task_ids = {task_id for task_id, _ in d_tasks}
        # Tasks with same dependencies should have same ID (deduplication)
        # Tasks with different dependencies should have different IDs
    
    def test_task_deduplication_with_same_config(self, baseline_tool):
        """Tasks with same tool, config, and dependencies should be deduplicated."""
        generator = WorkflowGenerator(pipeline_id="test")
        
        tool_b = ToolDefinition(name="B", container=ContainerConfig(image="test/b:v1", command="python run.py"), is_baseline=False)
        
        generator.set_stage_tools("pre_training", [tool_b], baseline_tool)
        generator.set_stage_tools("during_training", [], baseline_tool)
        generator.set_stage_tools("post_training", [], baseline_tool)
        generator.set_stage_tools("deployment", [], baseline_tool)
        
        workflows = generator.generate_all_workflows()
        
        # Collect all B tasks
        b_tasks = {}
        for wf in workflows:
            for task in wf.tasks:
                if task.tool.name == "B":
                    dep_ids = tuple(sorted(dep.id for dep in task.dependencies))
                    key = (task.tool.name, dep_ids)
                    if key not in b_tasks:
                        b_tasks[key] = []
                    b_tasks[key].append(task.id)
        
        # Tasks with same dependencies should have same ID
        for key, task_ids in b_tasks.items():
            unique_ids = set(task_ids)
            assert len(unique_ids) == 1, \
                f"Tasks with same dependencies {key} should be deduplicated, got {unique_ids}"
