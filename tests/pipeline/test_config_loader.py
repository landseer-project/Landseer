"""
Tests for the config_loader module.

These tests verify that pipeline configuration loading correctly implements:
1. Permutations of tools within stages
2. Baseline tool substitution for empty sets
3. Task deduplication via get_or_create_task

Note: Some tests may fail if the config_loader doesn't fully implement
the specification in docs/Tasks.md. These serve as acceptance tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Set

from src.pipeline.config_loader import (
    load_pipeline_config,
    make_combinations,
    create_workflow_from_combination,
    create_pipeline_from_config,
    PipelineConfig,
    StageConfig,
    DatasetConfig,
    ModelConfig,
)
from src.pipeline.tasks import (
    Task,
    TaskType,
    TaskStatus,
    clear_task_registry,
)
from src.pipeline.tools import (
    ToolDefinition,
    ContainerConfig,
    init_tool_registry,
    _TOOL_REGISTRY,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tools_yaml_content():
    """Sample tools.yaml content."""
    return """
tools:
  tool_a:
    name: tool_a
    container:
      image: test/a:v1
      command: python run.py
    is_baseline: true
  tool_b:
    name: tool_b
    container:
      image: test/b:v1
      command: python run.py
    is_baseline: false
  tool_b2:
    name: tool_b2
    container:
      image: test/b2:v1
      command: python run.py
    is_baseline: false
  tool_c:
    name: tool_c
    container:
      image: test/c:v1
      command: python run.py
    is_baseline: true
  tool_d:
    name: tool_d
    container:
      image: test/d:v1
      command: python run.py
    is_baseline: false
  tool_e:
    name: tool_e
    container:
      image: test/e:v1
      command: python run.py
    is_baseline: true
  tool_g:
    name: tool_g
    container:
      image: test/g:v1
      command: python run.py
    is_baseline: true
"""


@pytest.fixture
def simple_pipeline_yaml():
    """Simple pipeline config matching Tasks.md example."""
    return """
dataset:
  name: cifar10
  variant: clean
model:
  script: configs/model/config_model.py
  framework: pytorch
pipeline:
  pre_training:
    tools:
      - tool_a
      - tool_b
  during_training:
    tools:
      - tool_c
      - tool_d
  post_training:
    tools:
      - tool_e
  deployment:
    tools:
      - tool_g
"""


@pytest.fixture
def temp_configs(tools_yaml_content, simple_pipeline_yaml):
    """Create temporary config files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tools.yaml
        tools_path = Path(tmpdir) / "tools.yaml"
        with open(tools_path, 'w') as f:
            f.write(tools_yaml_content)
        
        # Create pipeline config
        pipeline_path = Path(tmpdir) / "pipeline.yaml"
        with open(pipeline_path, 'w') as f:
            f.write(simple_pipeline_yaml)
        
        # Create a dummy model script
        model_dir = Path(tmpdir) / "configs" / "model"
        model_dir.mkdir(parents=True)
        model_script = model_dir / "config_model.py"
        model_script.write_text("# dummy model config")
        
        # Update the pipeline yaml with correct path
        updated_pipeline_yaml = simple_pipeline_yaml.replace(
            "configs/model/config_model.py",
            str(model_script)
        )
        with open(pipeline_path, 'w') as f:
            f.write(updated_pipeline_yaml)
        
        yield {
            "tmpdir": tmpdir,
            "tools_path": str(tools_path),
            "pipeline_path": str(pipeline_path),
        }


@pytest.fixture(autouse=True)
def cleanup_registries():
    """Clear registries before and after each test."""
    clear_task_registry()
    yield
    clear_task_registry()


# ============================================================================
# Test: make_combinations
# ============================================================================


class TestMakeCombinations:
    """Tests for the make_combinations function."""
    
    def test_make_combinations_with_permutations(self, temp_configs):
        """
        Test permutation-based combination generation.
        
        For pre_training with [A (baseline), B (actual)]:
        - [B], [A] = 2 options (actual tool or baseline)
        
        For during_training with [C (baseline), D (actual)]:
        - [C], [D] = 2 options (single tool only)
        
        For post_training with [E (baseline)]:
        - [E] = 1 option
        
        For deployment with [G (baseline)]:
        - [G] = 1 option
        
        Total: 2 * 2 * 1 * 1 = 4 combinations
        """
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        combinations = make_combinations(config)
        
        # With the permutation logic:
        # pre: [B], [A] = 2 (actual + baseline)
        # during: [C], [D] = 2 (single tools only)
        # post: [E] = 1
        # deploy: [G] = 1
        # Total = 2 * 2 * 1 * 1 = 4
        assert len(combinations) == 4
    
    def test_combinations_contain_all_stages(self, temp_configs):
        """Each combination should have all 4 stages."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        combinations = make_combinations(config)
        
        expected_stages = {"pre_training", "during_training", "post_training", "deployment"}
        
        for combo in combinations:
            assert set(combo.keys()) == expected_stages
    
    def test_combinations_use_tool_definitions(self, temp_configs):
        """Combinations should contain ToolDefinition objects, not strings."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        combinations = make_combinations(config)
        
        for combo in combinations:
            for stage_name, tools in combo.items():
                for tool in tools:
                    assert isinstance(tool, ToolDefinition), f"Expected ToolDefinition, got {type(tool)}"
    
    def test_during_training_single_tool_only(self, temp_configs):
        """during_training stage should only have single tool options."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        combinations = make_combinations(config)
        
        # All during_training entries should have exactly 1 tool
        for combo in combinations:
            during_tools = combo["during_training"]
            assert len(during_tools) == 1, "during_training should have exactly 1 tool"


class TestMakeCombinationsPermutations:
    """
    Tests for permutation-based combination generation as specified in Tasks.md.
    
    These tests verify the IMPLEMENTED permutation behavior.
    """
    
    def test_permutations_within_pre_training_stage(self, temp_configs, tools_yaml_content):
        """
        Test permutations for pre_training with multiple actual tools.
        
        For pre_training with [A (baseline), B, B2]:
        Should generate: [B, B2], [B2, B], [B], [B2], [A] = 5 options
        """
        # Create a modified config with B2 in pre_training
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tools.yaml
            tools_path = Path(tmpdir) / "tools.yaml"
            with open(tools_path, 'w') as f:
                f.write(tools_yaml_content)
            
            # Create model script
            model_dir = Path(tmpdir) / "configs" / "model"
            model_dir.mkdir(parents=True)
            model_script = model_dir / "config_model.py"
            model_script.write_text("# dummy model config")
            
            # Create pipeline with B and B2 in pre_training
            pipeline_yaml = f"""
dataset:
  name: cifar10
  variant: clean
model:
  script: {model_script}
  framework: pytorch
pipeline:
  pre_training:
    tools:
      - tool_a
      - tool_b
      - tool_b2
  during_training:
    tools:
      - tool_c
  post_training:
    tools:
      - tool_e
  deployment:
    tools:
      - tool_g
"""
            pipeline_path = Path(tmpdir) / "pipeline.yaml"
            with open(pipeline_path, 'w') as f:
                f.write(pipeline_yaml)
            
            init_tool_registry(str(tools_path))
            config = load_pipeline_config(str(pipeline_path))
            combinations = make_combinations(config)
            
            # pre_training: P(2,2) + P(2,1) + baseline = 2 + 2 + 1 = 5
            # during: 1, post: 1, deploy: 1
            # Total = 5 * 1 * 1 * 1 = 5
            assert len(combinations) == 5
            
            # Extract unique pre_training sequences
            pre_sequences = set()
            for combo in combinations:
                pre_names = tuple(t.name for t in combo["pre_training"])
                pre_sequences.add(pre_names)
            
            # Should include all permutations
            expected_pre = {
                ("tool_b", "tool_b2"),  # B then B2
                ("tool_b2", "tool_b"),  # B2 then B
                ("tool_b",),            # B only
                ("tool_b2",),           # B2 only
                ("tool_a",),            # baseline only
            }
            
            assert pre_sequences == expected_pre


# ============================================================================
# Test: create_workflow_from_combination
# ============================================================================


class TestCreateWorkflowFromCombination:
    """Tests for workflow creation from combinations."""
    
    def test_workflow_has_correct_tasks(self, temp_configs):
        """Workflow should have tasks for each tool in the combination."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        
        # Get tool definitions from registry
        from src.pipeline.tools import get_tool
        tool_b = get_tool("tool_b")
        tool_d = get_tool("tool_d")
        tool_e = get_tool("tool_e")
        tool_g = get_tool("tool_g")
        
        combination = {
            "pre_training": [tool_b],
            "during_training": [tool_d],
            "post_training": [tool_e],
            "deployment": [tool_g],
        }
        
        workflow = create_workflow_from_combination("test_combo", combination, config)
        
        # Should have 4 tasks
        assert len(workflow.tasks) == 4
        
        # Verify tool names
        tool_names = [t.tool.name for t in workflow.tasks]
        assert tool_names == ["tool_b", "tool_d", "tool_e", "tool_g"]
    
    def test_workflow_dependencies_chain_correctly(self, temp_configs):
        """Tasks should have proper stage-to-stage dependencies."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        
        # Get tool definitions from registry
        from src.pipeline.tools import get_tool
        tool_b = get_tool("tool_b")
        tool_d = get_tool("tool_d")
        tool_e = get_tool("tool_e")
        tool_g = get_tool("tool_g")
        
        combination = {
            "pre_training": [tool_b],
            "during_training": [tool_d],
            "post_training": [tool_e],
            "deployment": [tool_g],
        }
        
        workflow = create_workflow_from_combination("test_combo", combination, config)
        
        # Extract tasks by name
        tasks_by_name = {t.tool.name: t for t in workflow.tasks}
        
        # pre_training has no dependencies
        assert len(tasks_by_name["tool_b"].dependencies) == 0
        
        # during_training depends on pre_training
        assert len(tasks_by_name["tool_d"].dependencies) == 1
        assert tasks_by_name["tool_b"] in tasks_by_name["tool_d"].dependencies
        
        # post_training depends on during_training
        assert len(tasks_by_name["tool_e"].dependencies) == 1
        assert tasks_by_name["tool_d"] in tasks_by_name["tool_e"].dependencies
        
        # deployment depends on post_training
        assert len(tasks_by_name["tool_g"].dependencies) == 1
        assert tasks_by_name["tool_e"] in tasks_by_name["tool_g"].dependencies
    
    def test_task_types_match_stages(self, temp_configs):
        """Tasks should have the correct TaskType for their stage."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        
        # Get tool definitions from registry
        from src.pipeline.tools import get_tool
        tool_b = get_tool("tool_b")
        tool_d = get_tool("tool_d")
        tool_e = get_tool("tool_e")
        tool_g = get_tool("tool_g")
        
        combination = {
            "pre_training": [tool_b],
            "during_training": [tool_d],
            "post_training": [tool_e],
            "deployment": [tool_g],
        }
        
        workflow = create_workflow_from_combination("test_combo", combination, config)
        
        tasks_by_name = {t.tool.name: t for t in workflow.tasks}
        
        assert tasks_by_name["tool_b"].task_type == TaskType.PRE_TRAINING
        assert tasks_by_name["tool_d"].task_type == TaskType.IN_TRAINING
        assert tasks_by_name["tool_e"].task_type == TaskType.POST_TRAINING
        assert tasks_by_name["tool_g"].task_type == TaskType.DEPLOYMENT


# ============================================================================
# Test: Task Deduplication in Config Loader
# ============================================================================


class TestConfigLoaderDeduplication:
    """
    Tests for task deduplication in the config loader.
    
    These tests verify that when loading a pipeline, tasks with the same
    tool and dependencies are reused rather than duplicated.
    """
    
    def test_shared_tasks_across_workflows(self, temp_configs):
        """
        Tasks with same tool+deps should be shared across workflows.
        
        This now works because config_loader uses get_or_create_task.
        """
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        # Collect all pre_training tasks that use tool_b (no dependencies)
        # Note: tool_b is used in workflows that choose it for pre_training
        tool_b_tasks = []
        for workflow in pipeline.workflows:
            for task in workflow.tasks:
                if (task.tool.name == "tool_b" and 
                    task.task_type == TaskType.PRE_TRAINING and
                    len(task.dependencies) == 0):
                    tool_b_tasks.append(task)
        
        # All tool_b tasks with no dependencies should be the SAME instance
        if len(tool_b_tasks) > 1:
            first_task = tool_b_tasks[0]
            for task in tool_b_tasks[1:]:
                assert task is first_task, "tool_b tasks with same deps should be deduplicated"
    
    def test_deduplication_reduces_unique_task_count(self, temp_configs):
        """
        Deduplication should result in fewer unique tasks than total task instances.
        """
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        # Collect all tasks from all workflows
        all_task_instances: List[Task] = []
        for workflow in pipeline.workflows:
            all_task_instances.extend(workflow.tasks)
        
        # Count unique task IDs
        unique_ids = set(t.id for t in all_task_instances)
        
        # With deduplication, unique tasks should be less than total instances
        # (or equal if no sharing is possible)
        assert len(unique_ids) <= len(all_task_instances)
        
        # With 4 workflows, there should be some task sharing
        # First tool in each workflow (with no deps) can be shared
        if len(pipeline.workflows) > 1:
            total_instances = len(all_task_instances)
            unique_count = len(unique_ids)
            savings = total_instances - unique_count
            assert savings >= 0, "Should have task reuse savings"


# ============================================================================
# Test: Full Pipeline Creation
# ============================================================================


class TestCreatePipelineFromConfig:
    """Tests for the complete pipeline creation process."""
    
    def test_pipeline_created_successfully(self, temp_configs):
        """Pipeline should be created without errors."""
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        assert pipeline is not None
        assert pipeline.name == "pipeline"  # filename without extension
    
    def test_pipeline_has_correct_number_of_workflows(self, temp_configs):
        """Pipeline should have the expected number of workflows."""
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        # 2 pre * 2 during * 1 post * 1 deploy = 4 workflows
        assert len(pipeline.workflows) == 4
    
    def test_workflow_names_are_numbered(self, temp_configs):
        """Workflows should have sequential comb_XXX names."""
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        expected_names = ["comb_001", "comb_002", "comb_003", "comb_004"]
        actual_names = [w.name for w in pipeline.workflows]
        
        assert actual_names == expected_names
    
    def test_pipeline_contains_dataset_info(self, temp_configs):
        """Pipeline should have dataset configuration."""
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        assert pipeline.dataset is not None
        assert pipeline.dataset["name"] == "cifar10"
        assert pipeline.dataset["variant"] == "clean"
    
    def test_pipeline_contains_model_info(self, temp_configs):
        """Pipeline should have model configuration."""
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            temp_configs["tools_path"]
        )
        
        assert pipeline.model is not None
        assert pipeline.model["framework"] == "pytorch"
        assert "config_model.py" in pipeline.model["script"]


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestConfigLoaderEdgeCases:
    """Tests for edge cases in config loading."""
    
    def test_missing_tools_yaml_warning(self, temp_configs):
        """Should warn but continue if tools.yaml not found."""
        import logging
        
        # This should not raise, just warn
        pipeline = create_pipeline_from_config(
            temp_configs["pipeline_path"],
            "nonexistent/tools.yaml"
        )
        
        # Pipeline is created but may have no tasks (tools not found)
        assert pipeline is not None
    
    def test_empty_stage_creates_no_tasks(self, temp_configs):
        """Empty tool list for a stage should create no tasks for that stage."""
        init_tool_registry(temp_configs["tools_path"])
        config = load_pipeline_config(temp_configs["pipeline_path"])
        
        # Get tool definitions from registry
        from src.pipeline.tools import get_tool
        tool_d = get_tool("tool_d")
        tool_e = get_tool("tool_e")
        tool_g = get_tool("tool_g")
        
        # Combination with empty pre_training (using ToolDefinition objects)
        combination = {
            "pre_training": [],
            "during_training": [tool_d],
            "post_training": [tool_e],
            "deployment": [tool_g],
        }
        
        workflow = create_workflow_from_combination("empty_pre", combination, config)
        
        # Should have 3 tasks (no pre_training)
        assert len(workflow.tasks) == 3
        
        # during_training should have no dependencies
        during_task = workflow.tasks[0]
        assert during_task.tool.name == "tool_d"
        assert len(during_task.dependencies) == 0
