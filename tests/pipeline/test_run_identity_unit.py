"""
Run-scoped task/workflow identity (src/pipeline/run_identity.py).

Tests cover:
- short_run_id: uses the run ID's hex suffix, hashes other shapes
- bind_pipeline_to_run: ID format, uniqueness, shared tasks keep one ID,
  workflow membership remapped, use_cache propagation
- lineage_hash: deterministic across builds, sensitive to config and
  dependencies, ignores run-scoped config keys
- Two runs of the same config no longer collide in the shared scheduler
"""

import pytest

from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.run_identity import (
    RUN_SCOPED_TASK_ID_RE,
    bind_pipeline_to_run,
    lineage_hash,
    short_run_id,
)
from src.pipeline.tasks import EvaluationTask, TaskFactory, TaskType, clear_task_registry
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from src.backend.scheduler.priority_scheduler import PriorityScheduler

RUN_A = "run_20261013_232027_e807ab0e"
RUN_B = "run_20261013_232028_3b81bf4a"


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _tool(name: str, image: str = "test/img:v1") -> ToolDefinition:
    return ToolDefinition(name=name, container=ContainerConfig(image=image, command="python main.py"))


def _build(pre_config=None, in_tools=("in-a", "in-b")) -> DefenseEvaluationPipeline:
    """
    One pre-training task shared by one workflow per in-training tool, plus an
    evaluation task per workflow (mirrors what the config loader builds).
    """
    pre = TaskFactory.create_task(
        task_type=TaskType.PRE_TRAINING,
        tool=_tool("noop"),
        config=dict(pre_config or {"stage": "pre_training", "tool_name": "noop"}),
    )
    pipeline = DefenseEvaluationPipeline(name="p", workflows=[])
    for idx, name in enumerate(in_tools, start=1):
        train = TaskFactory.create_task(
            task_type=TaskType.IN_TRAINING,
            tool=_tool(name),
            config={"stage": "in_training", "tool_name": name},
            dependencies=[pre],
        )
        ev = EvaluationTask(
            tool=_tool("clean"),
            config={"stage": "evaluation", "evaluator": "clean"},
            dependencies=[train],
        )
        wf = WorkflowFactory.create_workflow(name=f"comb_{idx:03d}", pipeline_id=pipeline.id)
        for t in (pre, train, ev):
            wf.add_task(t)
        pipeline.add_workflow(wf)
    return pipeline


def _all_tasks(pipeline):
    return {id(t): t for wf in pipeline.workflows for t in wf.tasks}.values()


# ----------------------------------------------------------------------------
# short_run_id
# ----------------------------------------------------------------------------


def test_short_run_id_uses_hex_suffix():
    assert short_run_id(RUN_A) == "e807ab0e"


def test_short_run_id_hashes_other_shapes():
    s = short_run_id("pipeline_3")
    assert len(s) == 8 and all(c in "0123456789abcdef" for c in s)
    assert s == short_run_id("pipeline_3")
    assert s != short_run_id("pipeline_4")


# ----------------------------------------------------------------------------
# bind_pipeline_to_run
# ----------------------------------------------------------------------------


def test_bind_assigns_run_scoped_unique_ids():
    p = _build()
    bind_pipeline_to_run(p, RUN_A)
    tasks = list(_all_tasks(p))
    assert len(tasks) == 5  # 1 shared pre + 2 train + 2 eval
    ids = [t.id for t in tasks]
    assert len(set(ids)) == len(ids)
    for t in tasks:
        assert RUN_SCOPED_TASK_ID_RE.match(t.id), t.id
        assert t.id.startswith("e807ab0e-")
        assert t.id[9:] == t.lineage_hash[:12]
        assert t.run_id == RUN_A and t.pipeline_id == RUN_A
    assert p.id == RUN_A


def test_shared_task_keeps_one_id_and_remapped_workflows():
    p = _build()
    bind_pipeline_to_run(p, RUN_A)
    wf_ids = {wf.id for wf in p.workflows}
    assert wf_ids == {"e807ab0e-comb_001", "e807ab0e-comb_002"}
    pre_1, pre_2 = (wf.tasks[0] for wf in p.workflows)
    assert pre_1 is pre_2
    assert pre_1.workflows == wf_ids
    for wf in p.workflows:
        assert wf.run_id == RUN_A and wf.pipeline_id == RUN_A
        for t in wf.tasks:
            assert t.workflows <= wf_ids


def test_same_config_in_two_runs_gets_disjoint_ids_with_same_lineage():
    p1 = _build()
    bind_pipeline_to_run(p1, RUN_A)
    clear_task_registry()
    p2 = _build()
    bind_pipeline_to_run(p2, RUN_B)
    ids1 = {t.id for t in _all_tasks(p1)}
    ids2 = {t.id for t in _all_tasks(p2)}
    assert ids1.isdisjoint(ids2)
    assert {i[9:] for i in ids1} == {i[9:] for i in ids2}


def test_use_cache_propagated_only_when_given():
    p = _build()
    bind_pipeline_to_run(p, RUN_A, use_cache=False)
    assert all(t.config["_run_use_cache"] is False for t in _all_tasks(p))
    clear_task_registry()
    p2 = _build()
    bind_pipeline_to_run(p2, RUN_B)
    assert all("_run_use_cache" not in t.config for t in _all_tasks(p2))


def test_use_cache_does_not_change_ids():
    p1 = _build()
    bind_pipeline_to_run(p1, RUN_A, use_cache=True)
    clear_task_registry()
    p2 = _build()
    bind_pipeline_to_run(p2, RUN_A, use_cache=False)
    assert {t.id for t in _all_tasks(p1)} == {t.id for t in _all_tasks(p2)}


# ----------------------------------------------------------------------------
# lineage_hash
# ----------------------------------------------------------------------------


def test_lineage_changes_with_upstream_config():
    p1 = _build(pre_config={"stage": "pre_training", "tool_name": "noop", "x": 1})
    clear_task_registry()
    p2 = _build(pre_config={"stage": "pre_training", "tool_name": "noop", "x": 2})
    # Same in-training tool, different upstream: lineage must differ.
    train1 = p1.workflows[0].tasks[1]
    train2 = p2.workflows[0].tasks[1]
    assert train1.config == train2.config
    assert lineage_hash(train1) != lineage_hash(train2)


def test_lineage_changes_with_image():
    a = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=_tool("t", "img:v1"), config={})
    b = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=_tool("t", "img:v2"), config={})
    assert lineage_hash(a) != lineage_hash(b)


def test_lineage_ignores_run_scoped_config():
    a = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=_tool("t"), config={"k": 1})
    b = TaskFactory.create_task(
        task_type=TaskType.PRE_TRAINING, tool=_tool("t"), config={"k": 1, "_run_use_cache": False}
    )
    assert lineage_hash(a) == lineage_hash(b)


# ----------------------------------------------------------------------------
# Scheduler: concurrent runs of the same config
# ----------------------------------------------------------------------------


def test_scheduler_accepts_two_runs_of_same_config():
    p1 = _build()
    bind_pipeline_to_run(p1, RUN_A)
    clear_task_registry()
    p2 = _build()
    bind_pipeline_to_run(p2, RUN_B)

    sched = PriorityScheduler(p1)
    added = sched.add_pipeline(p2)
    assert added == 5
    assert len(sched.get_all_tasks()) == 10
    assert {t.run_id for t in sched.get_all_tasks()} == {RUN_A, RUN_B}
