"""
Workflow generator unit tests.

Tests cover:
- generate_stage_permutations:
    * empty tool list with/without baseline
    * single tool with baseline
    * two tools with baseline: verifies all 5 options including permutations
    * three tools: count verification (3! + 3 + 1 = 10)
    * include_empty_as_baseline=False omits baseline option
- generate_during_training_options:
    * empty tools list with baseline falls back to [baseline]
    * single actual tool + baseline → 2 options
    * multiple tools → one option per tool + baseline
    * no tools, no baseline → single empty option
- WorkflowGenerator:
    * set_stage_tools rejects unknown stage
    * generate_all_stage_options respects during_training single-tool constraint
    * generate_workflow creates tasks with correct priorities
    * generate_all_workflows: task count is Cartesian product of stage options
    * task deduplication across workflows: shared tasks have counter > 1
    * get_workflow_summary returns correct unique task count
"""

import pytest
from typing import List

from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.tasks import (
    TaskType,
    TaskStatus,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.workflow_generator import (
    generate_stage_permutations,
    generate_during_training_options,
    WorkflowGenerator,
    StageTools,
)
import src.pipeline.tasks as tasks_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    clear_task_registry()
    tasks_module._task_id_counter = 0
    tasks_module._workflow_id_counter = 0
    tasks_module._pipeline_id_counter = 0
    yield
    clear_task_registry()


def make_tool(name: str, is_baseline: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=f"img/{name}:v1", command="run"),
        is_baseline=is_baseline,
    )


@pytest.fixture
def tool_b() -> ToolDefinition:
    return make_tool("tool_b")


@pytest.fixture
def tool_b2() -> ToolDefinition:
    return make_tool("tool_b2")


@pytest.fixture
def tool_b3() -> ToolDefinition:
    return make_tool("tool_b3")


@pytest.fixture
def baseline() -> ToolDefinition:
    return make_tool("noop", is_baseline=True)


@pytest.fixture
def in_tool_c() -> ToolDefinition:
    return make_tool("in_tool_c")


@pytest.fixture
def in_tool_d() -> ToolDefinition:
    return make_tool("in_tool_d")


# ============================================================================
# Tests: generate_stage_permutations
# ============================================================================


class TestGenerateStagePermutations:
    """Tests for generate_stage_permutations."""

    # -- empty tool list --

    def test_empty_tools_with_baseline_returns_baseline(self, baseline):
        # Arrange / Act
        result = generate_stage_permutations([], baseline=baseline)

        # Assert: only [[baseline]]
        assert result == [[baseline]]

    def test_empty_tools_without_baseline_returns_empty_seq(self):
        result = generate_stage_permutations([], baseline=None)
        assert result == [[]]

    def test_empty_tools_include_empty_false_returns_empty(self):
        result = generate_stage_permutations([], baseline=None, include_empty_as_baseline=False)
        assert result == [[]]

    # -- single tool --

    def test_single_tool_with_baseline(self, tool_b, baseline):
        # [B] → (B), (baseline)
        result = generate_stage_permutations([tool_b], baseline=baseline)
        assert [tool_b] in result
        assert [baseline] in result
        assert len(result) == 2

    def test_single_tool_no_baseline(self, tool_b):
        result = generate_stage_permutations([tool_b], baseline=None)
        assert result == [[tool_b]]

    # -- two tools --

    def test_two_tools_correct_count(self, tool_b, tool_b2, baseline):
        # [B, B2] → (B,B2), (B2,B), (B), (B2), (baseline) = 5 options
        result = generate_stage_permutations([tool_b, tool_b2], baseline=baseline)
        assert len(result) == 5

    def test_two_tools_contains_both_full_permutations(self, tool_b, tool_b2, baseline):
        result = generate_stage_permutations([tool_b, tool_b2], baseline=baseline)
        assert [tool_b, tool_b2] in result
        assert [tool_b2, tool_b] in result

    def test_two_tools_contains_singletons(self, tool_b, tool_b2, baseline):
        result = generate_stage_permutations([tool_b, tool_b2], baseline=baseline)
        assert [tool_b] in result
        assert [tool_b2] in result

    def test_two_tools_contains_baseline(self, tool_b, tool_b2, baseline):
        result = generate_stage_permutations([tool_b, tool_b2], baseline=baseline)
        assert [baseline] in result

    def test_two_tools_no_duplicates(self, tool_b, tool_b2, baseline):
        result = generate_stage_permutations([tool_b, tool_b2], baseline=baseline)
        # ToolDefinition is a Pydantic model (not hashable), compare as name-tuples
        as_name_tuples = [tuple(t.name for t in seq) for seq in result]
        assert len(as_name_tuples) == len(set(as_name_tuples))

    # -- three tools --

    def test_three_tools_correct_count(self, tool_b, tool_b2, tool_b3, baseline):
        # 3! + C(3,2)*2 + C(3,1) + 1(baseline)
        # = 6 + 6 + 3 + 1 = 16 ... wait:
        # permutations of 3: 6
        # permutations of 2 from 3: 3*2 = 6
        # permutations of 1 from 3: 3
        # baseline: 1
        # total: 16
        result = generate_stage_permutations([tool_b, tool_b2, tool_b3], baseline=baseline)
        assert len(result) == 16

    def test_three_tools_no_baseline_count(self, tool_b, tool_b2, tool_b3):
        # 6 + 6 + 3 = 15
        result = generate_stage_permutations([tool_b, tool_b2, tool_b3], baseline=None)
        assert len(result) == 15

    # -- include_empty_as_baseline=False --

    def test_exclude_baseline_option(self, tool_b, tool_b2, baseline):
        result = generate_stage_permutations(
            [tool_b, tool_b2], baseline=baseline, include_empty_as_baseline=False
        )
        # Should have 4 options, NOT 5 (no baseline)
        assert len(result) == 4
        assert [baseline] not in result


# ============================================================================
# Tests: generate_during_training_options
# ============================================================================


class TestGenerateDuringTrainingOptions:
    """Tests for generate_during_training_options."""

    # -- happy path --

    def test_single_tool_with_baseline(self, in_tool_c, baseline):
        result = generate_during_training_options([in_tool_c], baseline=baseline)
        # Expect [[in_tool_c], [baseline]]
        assert [in_tool_c] in result
        assert [baseline] in result
        assert len(result) == 2

    def test_two_tools_with_baseline(self, in_tool_c, in_tool_d, baseline):
        # [C, D, baseline] → 3 options, each single
        result = generate_during_training_options([in_tool_c, in_tool_d], baseline=baseline)
        assert [in_tool_c] in result
        assert [in_tool_d] in result
        assert [baseline] in result
        assert len(result) == 3

    def test_no_permutations_of_multiple_tools(self, in_tool_c, in_tool_d, baseline):
        # Must NOT produce [in_tool_c, in_tool_d] — only single-tool options
        result = generate_during_training_options([in_tool_c, in_tool_d], baseline=baseline)
        assert [in_tool_c, in_tool_d] not in result

    def test_each_result_has_exactly_one_tool(self, in_tool_c, in_tool_d, baseline):
        result = generate_during_training_options([in_tool_c, in_tool_d], baseline=baseline)
        for option in result:
            assert len(option) == 1

    # -- edge cases --

    def test_empty_tools_with_baseline(self, baseline):
        result = generate_during_training_options([], baseline=baseline)
        assert [baseline] in result
        assert len(result) == 1

    def test_empty_tools_no_baseline_returns_empty_option(self):
        result = generate_during_training_options([], baseline=None)
        # Falls through to [[]]
        assert result == [[]]

    def test_no_baseline_only_actual_tools(self, in_tool_c, in_tool_d):
        result = generate_during_training_options([in_tool_c, in_tool_d], baseline=None)
        assert len(result) == 2
        assert [in_tool_c] in result
        assert [in_tool_d] in result


# ============================================================================
# Tests: WorkflowGenerator
# ============================================================================


class TestWorkflowGenerator:
    """Tests for WorkflowGenerator class."""

    # -- set_stage_tools --

    def test_set_stage_tools_rejects_unknown_stage(self, tool_b, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        with pytest.raises(ValueError, match="Unknown stage"):
            gen.set_stage_tools("invalid_stage", actual_tools=[tool_b])

    def test_set_stage_tools_accepts_valid_stages(self, tool_b, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        for stage in ["pre_training", "during_training", "post_training", "deployment"]:
            gen.set_stage_tools(stage, actual_tools=[tool_b], baseline_tool=baseline)

    # -- generate_all_stage_options --

    def test_during_training_options_single_tool_only(self, in_tool_c, in_tool_d, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools(
            "during_training",
            actual_tools=[in_tool_c, in_tool_d],
            baseline_tool=baseline
        )
        options = gen.generate_all_stage_options()

        for opt in options["during_training"]:
            assert len(opt) <= 1

    def test_pre_training_gets_permutations(self, tool_b, tool_b2, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools("pre_training", actual_tools=[tool_b, tool_b2], baseline_tool=baseline)
        options = gen.generate_all_stage_options()

        # Should have 5 options for 2 actual tools + baseline
        assert len(options["pre_training"]) == 5

    def test_unset_stage_defaults_to_empty_option(self):
        gen = WorkflowGenerator(pipeline_id="p1")
        # No stages set at all
        options = gen.generate_all_stage_options()
        # Each unset stage has [[]] (one empty option)
        for stage in ["pre_training", "during_training", "post_training", "deployment"]:
            assert options[stage] == [[]]

    # -- generate_workflow --

    def test_generate_workflow_creates_workflow_object(self, tool_b, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        combo = {"pre_training": [tool_b], "during_training": [], "post_training": [], "deployment": []}
        wf = gen.generate_workflow(combo, workflow_name="comb_001")

        assert wf.name == "comb_001"
        assert len(wf.tasks) == 1

    def test_pre_training_task_priority_100(self, tool_b):
        gen = WorkflowGenerator(pipeline_id="p1")
        combo = {
            "pre_training": [tool_b],
            "during_training": [],
            "post_training": [],
            "deployment": []
        }
        wf = gen.generate_workflow(combo, "wf")
        pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]

        assert pre_tasks[0].priority == 100

    def test_during_training_task_priority_90(self, tool_b, in_tool_c):
        gen = WorkflowGenerator(pipeline_id="p1")
        combo = {
            "pre_training": [tool_b],
            "during_training": [in_tool_c],
            "post_training": [],
            "deployment": []
        }
        wf = gen.generate_workflow(combo, "wf")
        in_tasks = [t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING]

        assert in_tasks[0].priority == 90

    def test_within_stage_chaining(self, tool_b, tool_b2):
        # First tool has no deps; second tool depends on first
        gen = WorkflowGenerator(pipeline_id="p1")
        combo = {
            "pre_training": [tool_b, tool_b2],
            "during_training": [],
            "post_training": [],
            "deployment": []
        }
        wf = gen.generate_workflow(combo, "wf")
        pre_tasks = [t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING]

        assert len(pre_tasks) == 2
        first, second = pre_tasks[0], pre_tasks[1]
        assert first in second.dependencies

    def test_cross_stage_dependency(self, tool_b, in_tool_c):
        gen = WorkflowGenerator(pipeline_id="p1")
        combo = {
            "pre_training": [tool_b],
            "during_training": [in_tool_c],
            "post_training": [],
            "deployment": []
        }
        wf = gen.generate_workflow(combo, "wf")
        in_task = next(t for t in wf.tasks if t.task_type == TaskType.IN_TRAINING)
        pre_task = next(t for t in wf.tasks if t.task_type == TaskType.PRE_TRAINING)

        assert pre_task in in_task.dependencies

    # -- generate_all_workflows --

    def test_single_tool_per_stage_one_workflow(self, tool_b, in_tool_c, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools("pre_training", actual_tools=[tool_b], baseline_tool=baseline)
        gen.set_stage_tools("during_training", actual_tools=[in_tool_c], baseline_tool=baseline)
        # Only pre has [tool_b], [baseline] → 2 options
        # during has [in_tool_c], [baseline] → 2 options
        # post, deploy unset → [[]] each
        # total = 2 * 2 * 1 * 1 = 4
        workflows = gen.generate_all_workflows()
        assert len(workflows) == 4

    def test_task_deduplication_counter(self, tool_b, baseline, in_tool_c):
        # Arrange: pre_training with 1 actual tool and baseline
        # during_training with 1 actual tool and baseline
        # So 2*2 = 4 workflows; tool_b (at pre) appears in 2 of them with same deps
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools("pre_training", actual_tools=[tool_b], baseline_tool=baseline)
        gen.set_stage_tools("during_training", actual_tools=[in_tool_c], baseline_tool=baseline)
        workflows = gen.generate_all_workflows()

        # Collect all unique tasks
        all_tasks_by_id: dict = {}
        for wf in workflows:
            for task in wf.tasks:
                all_tasks_by_id[task.id] = task

        # At least one task should be reused (counter > 1)
        max_counter = max(t.counter for t in all_tasks_by_id.values())
        assert max_counter > 1

    def test_all_workflows_have_unique_ids(self, tool_b, in_tool_c, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools("pre_training", actual_tools=[tool_b], baseline_tool=baseline)
        gen.set_stage_tools("during_training", actual_tools=[in_tool_c], baseline_tool=baseline)
        workflows = gen.generate_all_workflows()

        wf_ids = [wf.id for wf in workflows]
        assert len(wf_ids) == len(set(wf_ids))

    # -- get_workflow_summary --

    def test_summary_unique_tasks_less_than_total(self, tool_b, in_tool_c, baseline):
        gen = WorkflowGenerator(pipeline_id="p1")
        gen.set_stage_tools("pre_training", actual_tools=[tool_b], baseline_tool=baseline)
        gen.set_stage_tools("during_training", actual_tools=[in_tool_c], baseline_tool=baseline)
        workflows = gen.generate_all_workflows()
        summary = gen.get_workflow_summary(workflows)

        assert summary["total_workflows"] == len(workflows)
        assert summary["total_unique_tasks"] <= summary["total_task_instances"]
        assert summary["task_reuse_savings"] >= 0

    def test_summary_empty_workflows(self):
        gen = WorkflowGenerator(pipeline_id="p1")
        summary = gen.get_workflow_summary([])
        assert summary["total_workflows"] == 0
        assert summary["total_unique_tasks"] == 0
        assert summary["total_task_instances"] == 0
