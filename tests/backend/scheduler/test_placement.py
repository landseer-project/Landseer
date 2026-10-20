"""Tool placement pins (configs/placement.yaml) and the scheduler accept filter."""

import os
import time
from pathlib import Path

import pytest

from src.pipeline.tasks import TaskFactory, TaskType, clear_task_registry
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import Workflow
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.scheduler.priority_scheduler import PriorityScheduler
from src.backend.scheduler.placement import Placement


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, container=ContainerConfig(image=f"img/{name}:v1", command="run"))


def _task(name: str):
    return TaskFactory.create_task(TaskType.DEPLOYMENT, tool=_tool(name))


def _write(path: Path, text: str) -> None:
    path.write_text(text)
    # Make the mtime change visible even on coarse-grained filesystems.
    t = time.time() + 5
    os.utime(path, (t, t))


def test_pinned_tool_goes_only_to_matching_workers(tmp_path):
    f = tmp_path / "placement.yaml"
    _write(f, "pins:\n  deploy_explainshap_celeba: [simurgh-]\n")
    p = Placement(str(f))
    shap, noop = _task("deploy_explainshap_celeba"), _task("deploy_noop")

    assert p.accepts("simurgh-gpu0")(shap)
    assert not p.accepts("gilbreth-11984977")(shap)
    assert not p.accepts("fabiha")(shap)
    assert not p.accepts(None)(shap)
    assert p.accepts("gilbreth-11984977")(noop)
    assert p.accepts(None)(noop)


def test_keys_match_registry_key_or_display_name(tmp_path):
    f = tmp_path / "placement.yaml"
    _write(f, "pins:\n  in_trades_celeba: simurgh-\n")
    assert not Placement(str(f)).accepts("fabiha")(_task("in-trades-celeba"))


def test_file_is_reloaded_when_it_changes(tmp_path):
    f = tmp_path / "placement.yaml"
    _write(f, "pins:\n  deploy_explainshap_celeba: [simurgh-]\n")
    p = Placement(str(f))
    shap = _task("deploy_explainshap_celeba")
    assert not p.accepts("fabiha")(shap)

    _write(f, "pins:\n  deploy_explainshap_celeba: [simurgh-, fabiha]\n")
    assert p.accepts("fabiha")(shap)


def test_missing_or_broken_file_pins_nothing(tmp_path):
    shap = _task("deploy_explainshap_celeba")
    assert Placement(str(tmp_path / "absent.yaml")).accepts("fabiha")(shap)

    f = tmp_path / "placement.yaml"
    _write(f, "pins: [unbalanced\n")
    assert Placement(str(f)).accepts("fabiha")(shap)


def test_scheduler_skips_rejected_task_for_next_one():
    shap, noop = _task("deploy_explainshap_celeba"), _task("deploy_noop")
    shap.priority, noop.priority = 100, 10
    wf = Workflow(name="wf", pipeline_id="")
    wf.add_task(shap)
    wf.add_task(noop)
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[])
    wf.pipeline_id = pipeline.id
    for t in wf.tasks:
        t.add_to_workflow(wf.id, pipeline.id)
    pipeline.add_workflow(wf)
    scheduler = PriorityScheduler(pipeline)

    not_shap = lambda t: t.tool.name != "deploy_explainshap_celeba"
    assert scheduler.get_next_task(accept=not_shap) is noop
    # Nothing else acceptable: the pinned task stays pending for its worker.
    assert scheduler.get_next_task(accept=not_shap) is None
    assert scheduler.get_next_task() is shap


def test_repo_placement_pins_explainshap_to_simurgh():
    p = Placement("configs/placement.yaml")
    shap = _task("deploy_explainshap_celeba")
    assert p.accepts("simurgh-gpu1")(shap)
    assert not p.accepts("gilbreth-11984977")(shap)
