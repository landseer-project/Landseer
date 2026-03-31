"""
Core behavior tests for Pipeline and DefenseEvaluationPipeline.

Focus:
- pipeline/workflow/task identity propagation
- single-workflow execution behavior
- failure isolation between workflows
- pipeline factory validation
"""

import pytest

from src.pipeline.pipeline import DefenseEvaluationPipeline, PipelineFactory
from src.pipeline.tasks import TaskFactory, TaskType, clear_task_registry
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory


@pytest.fixture(autouse=True)
def _reset_task_registry():
    clear_task_registry()
    yield
    clear_task_registry()


@pytest.fixture
def sample_tool():
    return ToolDefinition(
        name="pipeline_test_tool",
        container=ContainerConfig(image="test/tool:latest", command="python run.py"),
    )


def _make_task(sample_tool, name: str):
    return TaskFactory.create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=sample_tool,
        config={"name": name},
    )


def test_pipeline_post_init_sets_pipeline_id_on_existing_workflow_tasks(sample_tool):
    t1 = _make_task(sample_tool, "t1")
    wf = WorkflowFactory.create_workflow(name="wf_1", tasks=[t1])

    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf])

    assert wf.pipeline_id == pipeline.id
    assert t1.pipeline_id == pipeline.id
    assert wf.id in t1.workflows


def test_add_workflow_sets_workflow_pipeline_id(sample_tool):
    t1 = _make_task(sample_tool, "t1")
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[])
    wf = WorkflowFactory.create_workflow(name="wf_1", tasks=[t1])

    pipeline.add_workflow(wf)

    assert wf.pipeline_id == pipeline.id
    assert wf in pipeline.workflows


def test_get_workflow_by_name_returns_expected_workflow(sample_tool):
    wf_a = WorkflowFactory.create_workflow(name="comb_001", tasks=[_make_task(sample_tool, "a")])
    wf_b = WorkflowFactory.create_workflow(name="comb_002", tasks=[_make_task(sample_tool, "b")])
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf_a, wf_b])

    assert pipeline.get_workflow("comb_002").id == wf_b.id
    assert pipeline.get_workflow("missing") is None


def test_run_collects_success_and_failure_per_workflow(sample_tool):
    ok_wf = WorkflowFactory.create_workflow(name="ok", tasks=[_make_task(sample_tool, "ok")])

    class _BoomWorkflow:
        name = "boom"

        def run(self, _data=None):
            raise RuntimeError("workflow exploded")

    pipeline = DefenseEvaluationPipeline(name="p", workflows=[ok_wf, _BoomWorkflow()])
    results = pipeline.run(data={"input": 1})

    assert results["ok"]["status"] == "success"
    assert "result" in results["ok"]
    assert results["boom"]["status"] == "failed"
    assert "workflow exploded" in results["boom"]["error"]


def test_run_single_workflow_executes_only_requested_workflow(sample_tool):
    wf_a = WorkflowFactory.create_workflow(name="wf_a", tasks=[_make_task(sample_tool, "a")])
    wf_b = WorkflowFactory.create_workflow(name="wf_b", tasks=[_make_task(sample_tool, "b")])
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf_a, wf_b])

    result = pipeline.run_single_workflow("wf_b", data={"k": "v"})
    assert result == {"k": "v"}


def test_run_single_workflow_raises_with_available_names(sample_tool):
    wf_a = WorkflowFactory.create_workflow(name="wf_a", tasks=[_make_task(sample_tool, "a")])
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf_a])

    with pytest.raises(ValueError) as exc:
        pipeline.run_single_workflow("wf_missing")

    msg = str(exc.value)
    assert "wf_missing" in msg
    assert "wf_a" in msg


def test_pipeline_factory_creates_defense_pipeline(sample_tool):
    wf = WorkflowFactory.create_workflow(name="wf", tasks=[_make_task(sample_tool, "a")])
    pipeline = PipelineFactory.create_pipeline(
        name="factory_pipeline",
        workflows=[wf],
        config={"k": "v"},
        pipeline_type="defense_evaluation",
    )

    assert isinstance(pipeline, DefenseEvaluationPipeline)
    assert pipeline.name == "factory_pipeline"
    assert pipeline.config["k"] == "v"


def test_pipeline_factory_rejects_unknown_type():
    with pytest.raises(ValueError) as exc:
        PipelineFactory.create_pipeline(name="bad", pipeline_type="unknown_type")
    assert "Unknown pipeline type" in str(exc.value)


def test_shared_task_across_workflows_registered_once_with_both_workflow_ids(sample_tool):
    # Use same task object in two workflows for same pipeline
    shared = _make_task(sample_tool, "shared")
    wf1 = WorkflowFactory.create_workflow(name="wf_1", tasks=[shared])
    wf2 = WorkflowFactory.create_workflow(name="wf_2", tasks=[shared])

    pipeline = DefenseEvaluationPipeline(name="p", workflows=[wf1, wf2])

    # task should still belong to one pipeline, but be linked to both workflows
    assert shared.pipeline_id == pipeline.id
    assert wf1.id in shared.workflows
    assert wf2.id in shared.workflows
    assert shared.counter == 2
