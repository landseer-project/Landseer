"""
Per-run datasets on the backend.

Covers:
- prepare_run_dataset prepares/publishes a dataset once and reuses it for later
  runs; concurrent starts of the same dataset prepare it once
- model scripts are recorded per run (content-addressed MinIO keys)
- GET /dataset, /dataset/manifest, /dataset/download-url with run_id serve that
  run's dataset; unknown run_id is 404; without run_id the legacy view is kept
- the restart route prepares the restarted run's dataset (it used to skip it,
  so restarted runs had no dataset identity and missed the cache)
- the start-route helper records each run's own dataset

No MinIO, containers or real dataset preparation: the manager and store are fakes.
"""

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app, _scheduler_state
from src.backend.run_datasets import RunDatasetRegistry, prepare_run_dataset
from src.common import dataset_manifest as dsm
from src.db import ExperimentRunStatus
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.pipeline.tasks import TaskFactory, TaskType
from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.workflow import WorkflowFactory
from tests.worker.test_dataset_sync import MemStore


def _make_dataset(d: Path, fill: bytes, n: int = 100) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for name, size in (("data.npy", n), ("labels.npy", 10), ("test_data.npy", 50), ("test_labels.npy", 5)):
        (d / name).write_bytes(fill * size)
    dsm.write_manifest(d, dsm.build_manifest(d, provenance={"name": d.parent.name}))
    return d


class FakeManager:
    """DatasetManager.prepare_dataset over pre-built dirs; counts calls."""

    def __init__(self, dirs, delay: float = 0.0):
        self.dirs = dirs
        self.delay = delay
        self.calls = []

    def prepare_dataset(self, name, variant="clean", poisoning=None, **params):
        self.calls.append((name, variant, poisoning, params))
        time.sleep(self.delay)
        d = self.dirs[name]
        m = dsm.load_manifest(d)
        info = {
            "name": name, "variant": variant, "output_dir": str(d), "poisoning": poisoning,
            "train_samples": 10, "test_samples": 5, "content_id": m["content_id"],
            "provenance_id": m["provenance"]["provenance_id"], "provenance": m["provenance"],
            "file_count": m["file_count"], "total_bytes": m["total_bytes"],
        }
        return SimpleNamespace(to_dict=lambda: dict(info))


@pytest.fixture
def world(tmp_path):
    cifar = _make_dataset(tmp_path / "data" / "cifar10" / "clean", b"\x00")
    celeba = _make_dataset(tmp_path / "data" / "celeba" / "clean", b"\x01", n=300)
    script_a = tmp_path / "model_a.py"
    script_b = tmp_path / "model_b.py"
    script_a.write_text("NET = 'resnet20'\n")
    script_b.write_text("NET = 'resnet18'\n")
    store = MemStore()
    store.config = SimpleNamespace(bucket="landseer-artifacts", endpoint="minio:9000")
    return SimpleNamespace(
        cifar=cifar, celeba=celeba, script_a=script_a, script_b=script_b, store=store,
        manager=FakeManager({"cifar10": cifar, "celeba": celeba}),
    )


@pytest.fixture(autouse=True)
def fresh_registry():
    _scheduler_state.run_datasets = RunDatasetRegistry()
    yield
    _scheduler_state.run_datasets = RunDatasetRegistry()


def _ds(name):
    return {"name": name, "variant": "clean", "params": {}}


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


def test_dataset_prepared_once_and_reused(world):
    reg = RunDatasetRegistry()
    a = prepare_run_dataset(reg, "run_a", _ds("cifar10"), str(world.script_a), world.manager, world.store)
    b = prepare_run_dataset(reg, "run_b", _ds("cifar10"), str(world.script_b), world.manager, world.store)
    c = prepare_run_dataset(reg, "run_c", _ds("celeba"), str(world.script_a), world.manager, world.store)

    assert [call[0] for call in world.manager.calls] == ["cifar10", "celeba"]
    assert a["content_id"] == b["content_id"] != c["content_id"]
    assert a["minio_key"] == "datasets/cifar10/clean"
    assert c["minio_key"] == "datasets/celeba/clean"
    # Same dataset, different model scripts: separate, content-addressed keys.
    assert a["model_script_minio_key"] != b["model_script_minio_key"]
    assert a["model_script_minio_key"].startswith("datasets/cifar10/clean/models/")
    assert world.store.objects[a["model_script_minio_key"]] == world.script_a.read_bytes()
    assert world.store.objects[b["model_script_minio_key"]] == world.script_b.read_bytes()
    assert reg.get_run("run_a")["model_script"] == str(world.script_a)
    assert reg.get_run("run_b")["model_script_sha256"] != reg.get_run("run_a")["model_script_sha256"]
    # The dataset files were published once.
    assert world.store.uploads.count("datasets/cifar10/clean/dataset_manifest.json") == 1


def test_concurrent_starts_prepare_once(world):
    world.manager.delay = 0.2
    reg = RunDatasetRegistry()
    results = {}

    def _start(run_id):
        results[run_id] = prepare_run_dataset(
            reg, run_id, _ds("cifar10"), None, world.manager, world.store
        )

    threads = [threading.Thread(target=_start, args=(f"run_{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(world.manager.calls) == 1
    assert {r["content_id"] for r in results.values()} == {results["run_0"]["content_id"]}
    assert world.store.uploads.count("datasets/cifar10/clean/dataset_manifest.json") == 1


def test_poisoning_params_are_passed_once(world):
    reg = RunDatasetRegistry()
    ds = {"name": "cifar10", "variant": "poisoned", "params": {"poisoning": {"rate": 0.1}, "seed": 3}}
    prepare_run_dataset(reg, "run_p", ds, None, world.manager)
    assert world.manager.calls == [("cifar10", "poisoned", {"rate": 0.1}, {"seed": 3})]


def test_failed_publish_is_retried_by_next_run(world):
    reg = RunDatasetRegistry()
    real = world.store.upload_file
    world.store.upload_file = lambda p, k, **kw: False
    prepare_run_dataset(reg, "run_a", _ds("cifar10"), None, world.manager, world.store)
    world.store.upload_file = real
    b = prepare_run_dataset(reg, "run_b", _ds("cifar10"), None, world.manager, world.store)
    assert len(world.manager.calls) == 2
    assert b["minio_key"] == "datasets/cifar10/clean"


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------


@pytest.fixture
def client(world):
    reg = _scheduler_state.run_datasets
    prepare_run_dataset(reg, "run_cifar", _ds("cifar10"), str(world.script_a), world.manager, world.store)
    prepare_run_dataset(reg, "run_celeba", _ds("celeba"), str(world.script_b), world.manager, world.store)
    ctx = SimpleNamespace(store=world.store, dataset_info=None, pipeline=None)
    with patch("src.backend.api.routes.dataset.get_backend_context", return_value=ctx):
        yield TestClient(app, raise_server_exceptions=False)


def test_dataset_info_per_run(client, world):
    a = client.get("/dataset", params={"run_id": "run_cifar"}).json()
    b = client.get("/dataset", params={"run_id": "run_celeba"}).json()
    assert a["available"] and b["available"]
    assert (a["name"], b["name"]) == ("cifar10", "celeba")
    assert a["run_id"] == "run_cifar"
    assert a["content_id"] == dsm.load_manifest(world.cifar)["content_id"]
    assert b["content_id"] == dsm.load_manifest(world.celeba)["content_id"]
    assert a["minio_available"] and a["minio_key"] == "datasets/cifar10/clean"
    assert a["model_script"] == str(world.script_a)
    assert b["model_script"] == str(world.script_b)
    assert a["model_script_minio_key"] != b["model_script_minio_key"]
    assert a["model_script_sha256"] and a["config"]["name"] == "cifar10"


def test_dataset_manifest_per_run(client, world):
    a = client.get("/dataset/manifest", params={"run_id": "run_cifar"})
    b = client.get("/dataset/manifest", params={"run_id": "run_celeba"})
    assert a.status_code == b.status_code == 200
    assert a.json()["content_id"] == dsm.load_manifest(world.cifar)["content_id"]
    assert b.json()["content_id"] == dsm.load_manifest(world.celeba)["content_id"]


def test_download_url_per_run(client):
    r = client.get("/dataset/download-url", params={"run_id": "run_celeba"})
    assert r.status_code == 200
    assert r.json()["minio_key"] == "datasets/celeba/clean"


@pytest.mark.parametrize("path", ["/dataset", "/dataset/manifest", "/dataset/download-url"])
def test_unknown_run_is_404(client, path):
    r = client.get(path, params={"run_id": "run_missing"})
    assert r.status_code == 404
    assert "run_missing" in r.json()["detail"]


def test_without_run_id_legacy_view_is_kept(client):
    # No legacy dataset_info on the context: same response as before.
    r = client.get("/dataset")
    assert r.status_code == 200
    assert r.json()["available"] is False
    assert client.get("/dataset/manifest").status_code == 404
    assert client.get("/dataset/download-url").status_code == 404


# ----------------------------------------------------------------------------
# Start / restart routes
# ----------------------------------------------------------------------------


def _pipeline(dataset: str, script: Path, name="p"):
    tool = ToolDefinition(name="test_tool", container=ContainerConfig(image="test/img:v1", command="run"))
    task = TaskFactory.create_task(task_type=TaskType.PRE_TRAINING, tool=tool, config={})
    wf = WorkflowFactory.create_workflow(name="wf_001", tasks=[task])
    return DefenseEvaluationPipeline(
        name=name, workflows=[wf], dataset=_ds(dataset), model={"script": str(script)}
    )


def _ctx(world):
    return SimpleNamespace(store=world.store, dataset_manager=world.manager, dataset_info=None, pipeline=None)


def test_start_helper_records_each_runs_dataset(world):
    from src.backend.api.routes.pipeline_configs import _prepare_run_dataset

    ctx = _ctx(world)
    with patch("src.backend.api.routes.pipeline_configs.get_backend_context", return_value=ctx), \
         patch("src.backend.api.routes.pipeline_configs.set_backend_context"):
        _prepare_run_dataset(_scheduler_state, "run_1", _pipeline("cifar10", world.script_a))
        _prepare_run_dataset(_scheduler_state, "run_2", _pipeline("celeba", world.script_b))

    reg = _scheduler_state.run_datasets
    assert reg.get_run("run_1")["name"] == "cifar10"
    assert reg.get_run("run_2")["name"] == "celeba"
    # Legacy single view: the latest run.
    assert ctx.dataset_info["run_id"] == "run_2"


@contextmanager
def _fake_session_scope():
    yield object()


def test_restart_prepares_the_runs_dataset(world):
    old_run = SimpleNamespace(
        id="run_old", experiment_id="config_celeba", run_number=1, use_cache=True,
        status=ExperimentRunStatus.COMPLETED,
    )

    class _Repo:
        def __init__(self, _s):
            pass

        def get_by_id(self, run_id):
            return old_run if run_id == "run_old" else None

        def get_next_run_number(self, _):
            return 2

        def create(self, data):
            return SimpleNamespace(
                id=data["id"], experiment_id=data["experiment_id"], run_number=data["run_number"],
                use_cache=data["use_cache"], tools_config=None, status=ExperimentRunStatus.PENDING,
                error_message=None, created_at=SimpleNamespace(isoformat=lambda: "2026-10-19T12:00:00"),
                started_at=None, completed_at=None,
            )

        def update_status(self, run_id, status, error_message=None):
            return None

    cfg = SimpleNamespace(id="config_celeba", name="celeba", config_path="/tmp/celeba.yaml",
                          attack_config_path=None)
    ctx = _ctx(world)
    client = TestClient(app, raise_server_exceptions=False)
    with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), \
         patch("src.db.session_scope", _fake_session_scope), \
         patch("src.db.ExperimentRunRepository", _Repo), \
         patch("src.pipeline.config_loader.create_pipeline_from_config",
               return_value=_pipeline("celeba", world.script_b)), \
         patch("src.backend.api.routes.pipeline_configs.get_backend_context", return_value=ctx), \
         patch("src.backend.api.routes.pipeline_configs.set_backend_context"):
        resp = client.post("/api/pipeline-runs/run_old/restart", json={"use_cache": True})
        assert resp.status_code == 200, resp.text
        new_id = resp.json()["id"]
        info = client.get("/dataset", params={"run_id": new_id}).json()

    assert resp.json()["status"] == ExperimentRunStatus.RUNNING.value
    assert info["name"] == "celeba"
    assert info["content_id"] == dsm.load_manifest(world.celeba)["content_id"]
    assert info["model_script"] == str(world.script_b)
    assert new_id in _scheduler_state.pipelines
