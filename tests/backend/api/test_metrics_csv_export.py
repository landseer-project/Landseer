"""
Unit tests for _export_run_metrics_csv.

Verifies CSV path, headers, status mapping, and -1 placeholders for
missing/failed/skipped evaluators.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.backend.api import _export_run_metrics_csv
from src.pipeline.tasks import TaskType, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline

from .conftest import create_task_with_id


RUN_ID = "run_metrics_csv_test_001"
CONFIG_ID = "config_metrics_csv_test"


def _eval_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image="eval/image:latest", command="python eval.py"),
    )


def _defense_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image="defense/image:latest", command="python main.py"),
    )


def _make_eval_task(task_id: str, tool_name: str, metrics: list[str]):
    task = create_task_with_id(
        task_id=task_id,
        tool=_eval_tool(tool_name),
        task_type=TaskType.EVALUATION,
        priority=50,
        pipeline_id=None,
        config={"metrics": metrics},
    )
    return task


def _make_defense_task(task_id: str, tool_name: str, task_type: TaskType, priority: int):
    return create_task_with_id(
        task_id=task_id,
        tool=_defense_tool(tool_name),
        task_type=task_type,
        priority=priority,
        pipeline_id=None,
        config={"stage": task_type.value, "tool_name": tool_name},
    )


def _make_pipeline():
    clear_task_registry()

    wf_ok = WorkflowFactory.create_workflow(
        name="ok_combo",
        tasks=[
            _make_defense_task("t_pre_z", "pre_z", TaskType.PRE_TRAINING, 100),
            _make_defense_task("t_pre_a", "pre_a", TaskType.PRE_TRAINING, 100),
            _make_defense_task("t_in_fair", "in_fair", TaskType.IN_TRAINING, 90),
            _make_defense_task("t_post_noop", "post_noop", TaskType.POST_TRAINING, 80),
            _make_defense_task("t_deploy_noop", "deploy_noop", TaskType.DEPLOYMENT, 70),
            _make_eval_task("t_ok_adv", "adversarial", ["clean_accuracy", "pgd_accuracy"]),
            _make_eval_task("t_ok_fair", "fairness", ["demographic_parity"]),
        ],
    )
    wf_missing = WorkflowFactory.create_workflow(
        name="missing_combo",
        tasks=[_make_eval_task("t_miss_fair", "fairness", ["demographic_parity"])],
    )
    wf_failed = WorkflowFactory.create_workflow(
        name="failed_combo",
        tasks=[_make_eval_task("t_fail_fair", "fairness", ["demographic_parity"])],
    )
    wf_skipped = WorkflowFactory.create_workflow(
        name="skipped_combo",
        tasks=[_make_eval_task("t_skip_fair", "fairness", ["demographic_parity"])],
    )

    wf_ok.id = "wf_ok"
    wf_missing.id = "wf_missing"
    wf_failed.id = "wf_failed"
    wf_skipped.id = "wf_skipped"

    pipeline = DefenseEvaluationPipeline(
        name="metrics_csv_pipeline",
        workflows=[wf_ok, wf_missing, wf_failed, wf_skipped],
    )
    pipeline.id = RUN_ID
    return pipeline


def _fake_result(workflow_id, evaluator_name, metrics, success=True, skipped=False):
    return SimpleNamespace(
        workflow_id=workflow_id,
        evaluator_name=evaluator_name,
        metrics=metrics,
        success=success,
        skipped=skipped,
    )


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, run, results):
        self._run = run
        self._results = results

    def query(self, model):
        name = getattr(model, "__name__", str(model))
        if "PipelineRun" in name:
            return _FakeQuery([self._run] if self._run else [])
        if "EvaluationResult" in name:
            return _FakeQuery(self._results)
        return _FakeQuery([])


@pytest.fixture
def tmp_results_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_export_run_metrics_csv_writes_expected_rows(tmp_results_cwd):
    pipeline = _make_pipeline()
    scheduler = SimpleNamespace(pipeline=pipeline)

    run = SimpleNamespace(id=RUN_ID, pipeline_config_id=CONFIG_ID)
    results = [
        _fake_result("wf_ok", "adversarial", {"clean_accuracy": 0.91, "pgd_accuracy": 0.7}),
        _fake_result("wf_ok", "fairness", {"demographic_parity": 0.12}),
        _fake_result("wf_failed", "fairness", {}, success=False),
        _fake_result("wf_skipped", "fairness", {}, success=True, skipped=True),
    ]

    fake_session = _FakeSession(run, results)

    @contextmanager
    def fake_scope():
        yield fake_session

    with patch("src.db.session_scope", fake_scope):
        csv_path = _export_run_metrics_csv(RUN_ID, scheduler)

    assert csv_path is not None
    expected = Path("results") / CONFIG_ID / RUN_ID / "metrics_summary.csv"
    assert csv_path == expected
    assert csv_path.exists()

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_wf = {r["workflow_id"]: r for r in rows}
    assert set(by_wf) == {"wf_failed", "wf_missing", "wf_ok", "wf_skipped"}

    ok = by_wf["wf_ok"]
    assert ok["run_id"] == RUN_ID
    assert ok["workflow_name"] == "ok_combo"
    assert ok["pre"] == "pre_z -> pre_a"
    assert ok["in"] == "in_fair"
    assert ok["post"] == "post_noop"
    assert ok["deploy"] == "deploy_noop"
    assert ok["pre_training"] == "pre_z -> pre_a"
    assert ok["in_training"] == "in_fair"
    assert ok["post_training"] == "post_noop"
    assert ok["deployment"] == "deploy_noop"
    assert ok["combination_success"] == "success"
    assert ok["adversarial.status"] == "ok"
    assert float(ok["adversarial.clean_accuracy"]) == pytest.approx(0.91)
    assert float(ok["adversarial.pgd_accuracy"]) == pytest.approx(0.7)
    assert ok["fairness.status"] == "ok"
    assert float(ok["fairness.demographic_parity"]) == pytest.approx(0.12)

    missing = by_wf["wf_missing"]
    assert missing["fairness.status"] == "missing"
    assert missing["fairness.demographic_parity"] == "-1"
    assert missing["adversarial.status"] == "not_applicable"
    assert missing["adversarial.clean_accuracy"] == "-1"
    assert missing["combination_success"] == "failure"

    failed = by_wf["wf_failed"]
    assert failed["fairness.status"] == "failed"
    assert failed["fairness.demographic_parity"] == "-1"
    assert failed["combination_success"] == "failure"

    skipped = by_wf["wf_skipped"]
    assert skipped["fairness.status"] == "skipped"
    assert skipped["fairness.demographic_parity"] == "-1"
    assert skipped["combination_success"] == "success"


def test_export_skips_when_no_evaluation_tasks(tmp_results_cwd):
    clear_task_registry()
    train = create_task_with_id(
        task_id="task_train",
        tool=ToolDefinition(
            name="in_noop",
            container=ContainerConfig(image="x:latest", command="true"),
        ),
        task_type=TaskType.IN_TRAINING,
        priority=10,
        pipeline_id=None,
    )
    wf = WorkflowFactory.create_workflow(name="no_eval", tasks=[train])
    pipeline = DefenseEvaluationPipeline(name="no_eval_pipe", workflows=[wf])
    scheduler = SimpleNamespace(pipeline=pipeline)

    csv_path = _export_run_metrics_csv(RUN_ID, scheduler)
    assert csv_path is None
    assert not list(Path("results").rglob("metrics_summary.csv")) if Path("results").exists() else True


def test_export_returns_none_without_pipeline(tmp_results_cwd):
    scheduler = SimpleNamespace(pipeline=None)
    assert _export_run_metrics_csv(RUN_ID, scheduler) is None
