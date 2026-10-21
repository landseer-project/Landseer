"""
Regression test: concurrent pipeline syncs against a file-backed SQLite DB.

On 2026-10-21 three run starts arrived within ~1 s; their
``sync_pipeline_to_db`` calls interleaved on the single shared sqlite3
connection (StaticPool) and failed with ``sqlite3.DatabaseError: not an
error``, leaving runs with partial or zero task rows.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from src.backend.db_service import DatabaseService
from src.db import (
    DatabaseConfig,
    ExperimentRepository,
    ExperimentRunRepository,
    ExperimentRunStatus,
    get_database,
    session_scope,
)
from src.db.models import TaskModel
from src.pipeline.tasks import TaskStatus as PipelineTaskStatus

N_RUNS = 6
TASKS_PER_RUN = 120


class _FakeTask:
    def __init__(self, task_id: str, tool, deps=()):
        self.id = task_id
        self.tool = tool
        self.task_type = "tool"
        self.config = {"i": task_id}
        self.priority = 0
        self.status = PipelineTaskStatus.PENDING
        self.counter = 0
        self.dependencies = list(deps)

    def get_hash(self) -> str:
        return f"hash-{self.id}"


def _tool(name: str):
    return SimpleNamespace(
        id=name,
        name=name,
        is_baseline=True,
        defense_stage="deployment",
        container=SimpleNamespace(image=f"img/{name}:v1"),
        parameters={},
    )


def _fake_pipeline(run_id: str):
    # Every run shares the same tools, as concurrent runs of real configs do.
    tools = [_tool("deploy_noop"), _tool("train_noop"), _tool("eval_noop")]
    workflows = []
    for w in range(TASKS_PER_RUN // 3):
        tasks = []
        prev = None
        for t, tool in enumerate(tools):
            task = _FakeTask(f"{run_id}_w{w}_t{t}", tool, deps=[prev] if prev else [])
            tasks.append(task)
            prev = task
        workflows.append(SimpleNamespace(id=f"{run_id}_w{w}", tasks=tasks))
    return SimpleNamespace(id=run_id, run_id=run_id, workflows=workflows)


@pytest.fixture
def db_service(tmp_path):
    svc = DatabaseService(
        DatabaseConfig(db_type="sqlite", sqlite_path=str(tmp_path / "concurrency.db"))
    )
    assert svc.initialize()
    yield svc
    get_database().close()


def test_concurrent_pipeline_syncs_persist_every_task(db_service):
    run_ids = [f"run_{i}" for i in range(N_RUNS)]
    with session_scope() as session:
        ExperimentRepository(session).create(
            {"id": "cfg", "name": "cfg", "config_path": "/tmp/cfg.yaml", "config_hash": "h"}
        )
        for n, run_id in enumerate(run_ids, start=1):
            ExperimentRunRepository(session).create(
                {
                    "id": run_id,
                    "experiment_id": "cfg",
                    "run_number": n,
                    "status": ExperimentRunStatus.PENDING,
                }
            )

    pipelines = [_fake_pipeline(r) for r in run_ids]
    barrier = threading.Barrier(N_RUNS + 1)
    results: dict = {}
    stop = threading.Event()

    def _sync(p):
        barrier.wait()
        results[p.id] = db_service.sync_pipeline_to_db(p)

    def _status_writer():
        # Concurrent short transactions, like worker status/heartbeat writes.
        barrier.wait()
        while not stop.is_set():
            with session_scope() as session:
                ExperimentRunRepository(session).update_status(
                    run_ids[0], ExperimentRunStatus.RUNNING
                )

    threads = [threading.Thread(target=_sync, args=(p,)) for p in pipelines]
    writer = threading.Thread(target=_status_writer)
    for t in threads + [writer]:
        t.start()
    for t in threads:
        t.join(timeout=120)
    stop.set()
    writer.join(timeout=30)

    assert results == {r: r for r in run_ids}
    with session_scope() as session:
        for run_id in run_ids:
            count = (
                session.query(TaskModel)
                .filter(TaskModel.experiment_run_id == run_id)
                .count()
            )
            assert count == TASKS_PER_RUN, f"{run_id}: {count}/{TASKS_PER_RUN} tasks"
