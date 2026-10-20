"""
Per-run datasets on the worker.

Runs on different datasets (a CIFAR-10 and a CelebA config) share workers, so
each task resolves the dataset and model script of its own run.

Covers:
- dataset info, manifest and model script are requested with the task's run_id
- LANDSEER_DATA_PATH is used only for the run whose manifest it matches; other
  runs fall back to the MinIO copy instead of being refused
- LANDSEER_DATA_ROOT/<name>/<variant> is preferred over a MinIO download when it
  matches, and skipped when it does not
- a run is refused only when its dataset cannot be had verified from anywhere
- switching between runs re-fetches and re-verifies nothing (per-run and
  per-dataset memo)
- cache keys differ across datasets and model scripts, and are stable within one
- model scripts are fetched by content hash from MinIO
- verify_dir: hashes size-mismatched files in one hash_files call; quick mode
  stops at the first size mismatch without hashing

No containers or GPUs: the runner and client are mocks, MinIO is in memory.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from src.backend.dataset_publish import publish_dataset
from src.common import dataset_manifest as dsm
from src.worker.cli import Worker
from src.worker.client import TaskInfo
from src.worker.runner import ExecutionResult
from tests.worker.test_dataset_sync import MemStore


def _make_dataset(d: Path, fill: bytes, n: int = 100) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.npy").write_bytes(fill * n)
    (d / "labels.npy").write_bytes(fill * 10)
    (d / "test_data.npy").write_bytes(fill * 50)
    (d / "test_labels.npy").write_bytes(fill * 5)
    return d


def _task(task_id: str, run_id: str) -> TaskInfo:
    return TaskInfo(
        id=task_id,
        tool_name="tool",
        tool_image="img:1",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={"p": 1},
        priority=1,
        status="pending",
        task_type="pre_training",
        counter=1,
        workflows=["w1"],
        pipeline_id=run_id,
        dependency_ids=[],
        run_id=run_id,
    )


class FakeBackend:
    """GET /dataset?run_id= and /dataset/manifest?run_id= for a few runs."""

    def __init__(self):
        self.infos: Dict[str, dict] = {}
        self.manifests: Dict[str, dict] = {}
        self.info_calls = []
        self.manifest_calls = []

    def add_run(self, run_id, name, src_dir: Path, store: MemStore, model_script: Path = None):
        dsm.write_manifest(src_dir, dsm.build_manifest(src_dir, provenance={"name": name}))
        info = {"name": name, "variant": "clean", "output_dir": str(src_dir)}
        assert publish_dataset(store, info, f"datasets/{name}/clean")
        manifest = dsm.load_manifest(src_dir)
        self.manifests[run_id] = manifest
        self.infos[run_id] = {
            "available": True,
            "run_id": run_id,
            "name": name,
            "variant": "clean",
            # Backend dir not visible from this worker: forces manual dirs / MinIO.
            "local_path": None,
            "minio_key": info["minio_key"],
            "manifest_minio_key": info["manifest_minio_key"],
            "minio_available": True,
            "content_id": manifest["content_id"],
        }
        if model_script is not None:
            sha = hashlib.sha256(model_script.read_bytes()).hexdigest()
            key = f"datasets/{name}/clean/models/{sha[:16]}/config_model.py"
            store.upload_file(model_script, key)
            self.infos[run_id].update(
                model_script=str(model_script), model_script_sha256=sha, model_script_minio_key=key
            )

    def get_dataset_info(self, run_id=None):
        self.info_calls.append(run_id)
        return dict(self.infos.get(run_id, {"available": False}))

    def get_dataset_manifest(self, run_id=None):
        self.manifest_calls.append(run_id)
        return self.manifests.get(run_id)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("LANDSEER_ALLOW_DATASET_MISMATCH", raising=False)
    monkeypatch.delenv("LANDSEER_PIPELINE_CONFIG", raising=False)
    monkeypatch.delenv("LANDSEER_DATA_ROOT", raising=False)
    store = MemStore()
    backend = FakeBackend()
    # The backend's prepared copies (not visible to the worker except via MinIO).
    cifar_src = _make_dataset(tmp_path / "backend" / "cifar10" / "clean", b"\x00")
    celeba_src = _make_dataset(tmp_path / "backend" / "celeba" / "clean", b"\x01", n=300)
    backend.add_run("run_cifar", "cifar10", cifar_src, store)
    backend.add_run("run_celeba", "celeba", celeba_src, store)
    return SimpleNamespace(tmp=tmp_path, store=store, backend=backend,
                           cifar_src=cifar_src, celeba_src=celeba_src)


def _worker(env, **kwargs) -> Worker:
    w = Worker(workspace_dir=env.tmp / "ws", cache_dir=env.tmp / "wcache", use_cache=False, **kwargs)
    w._client = MagicMock()
    w._client.get_dataset_info.side_effect = env.backend.get_dataset_info
    w._client.get_dataset_manifest.side_effect = env.backend.get_dataset_manifest
    w._client.get_pipeline_info.return_value = {}
    w._two_level_cache = SimpleNamespace(manager=SimpleNamespace(minio=env.store))
    w._runner = MagicMock()
    w._runner.workspace_dir = env.tmp / "ws"

    def _run_task(task, **kw):
        out = env.tmp / "ws" / task.id / "output"
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pt").write_bytes(b"m")
        return ExecutionResult(success=True, exit_code=0, execution_time_ms=1, output_path=out)

    w._runner.run_task.side_effect = _run_task
    return w


def _input_path(worker) -> Path:
    return worker._runner.run_task.call_args.kwargs["input_path"]


def _cache_key(worker, task_id) -> str:
    return worker._task_outputs[task_id][0]


# ----------------------------------------------------------------------------
# Choosing the run's dataset
# ----------------------------------------------------------------------------


def test_lookups_carry_the_tasks_run_id(env):
    w = _worker(env)
    assert w._execute_task(_task("t1", "run_cifar")).success
    assert env.backend.info_calls == ["run_cifar"]
    assert w._dataset_info["name"] == "cifar10"


def test_manual_path_used_for_matching_run_and_minio_for_the_other(env):
    manual = _make_dataset(env.tmp / "manual_celeba", b"\x01", n=300)
    w = _worker(env, data_path=manual)

    assert w._execute_task(_task("t1", "run_celeba")).success
    assert _input_path(w) == manual
    assert env.store.downloads == []

    # The CIFAR-10 run is not refused: its verified MinIO copy is used.
    result = w._execute_task(_task("t2", "run_cifar"))
    assert result.success
    synced = env.tmp / "wcache" / "datasets" / "cifar10" / "clean"
    assert _input_path(w) == synced
    assert (synced / "data.npy").read_bytes() == (env.cifar_src / "data.npy").read_bytes()
    assert result.artifacts["dataset_content_id"] == env.backend.manifests["run_cifar"]["content_id"]


def test_data_root_is_preferred_over_minio(env):
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x00")
    _make_dataset(root / "celeba" / "clean", b"\x01", n=300)
    w = _worker(env, data_root=root)

    assert w._execute_task(_task("t1", "run_cifar")).success
    assert _input_path(w) == root / "cifar10" / "clean"
    assert w._execute_task(_task("t2", "run_celeba")).success
    assert _input_path(w) == root / "celeba" / "clean"
    assert env.store.downloads == []


def test_data_root_from_env(env, monkeypatch):
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x00")
    monkeypatch.setenv("LANDSEER_DATA_ROOT", str(root))
    w = _worker(env)
    assert w.data_root == root
    assert w._execute_task(_task("t1", "run_cifar")).success
    assert _input_path(w) == root / "cifar10" / "clean"


def test_stale_data_root_copy_falls_back_to_minio(env):
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x09")  # same sizes, other content
    w = _worker(env, data_root=root)

    assert w._execute_task(_task("t1", "run_cifar")).success
    assert _input_path(w) == env.tmp / "wcache" / "datasets" / "cifar10" / "clean"


def test_data_path_wins_over_data_root(env):
    manual = _make_dataset(env.tmp / "manual", b"\x00")
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x00")
    w = _worker(env, data_path=manual, data_root=root)
    assert w._execute_task(_task("t1", "run_cifar")).success
    assert _input_path(w) == manual


def test_refused_only_when_no_verified_copy_exists(env):
    manual = _make_dataset(env.tmp / "manual_celeba", b"\x01", n=300)
    w = _worker(env, data_path=manual)
    w._two_level_cache = None  # no MinIO: the CIFAR-10 run cannot be fetched

    result = w._execute_task(_task("t1", "run_cifar"))
    assert not result.success
    assert "LANDSEER_DATA_PATH" in result.error_message
    assert "No verified copy" in result.error_message
    w._runner.run_task.assert_not_called()

    # The matching run on the same worker is unaffected.
    assert w._execute_task(_task("t2", "run_celeba")).success
    assert _input_path(w) == manual


def test_override_is_a_last_resort(env, monkeypatch):
    monkeypatch.setenv("LANDSEER_ALLOW_DATASET_MISMATCH", "1")
    manual = _make_dataset(env.tmp / "manual_celeba", b"\x01", n=300)
    w = _worker(env, data_path=manual)

    # A verified copy exists: the override does not force the wrong dataset.
    assert w._execute_task(_task("t1", "run_cifar")).success
    assert _input_path(w) == env.tmp / "wcache" / "datasets" / "cifar10" / "clean"

    w2 = _worker(env, data_path=manual)
    w2._two_level_cache = None
    result = w2._execute_task(_task("t1", "run_cifar"))
    assert result.success and _input_path(w2) == manual
    assert result.artifacts["dataset_content_id"] != env.backend.manifests["run_cifar"]["content_id"]


# ----------------------------------------------------------------------------
# Memoization across run switches
# ----------------------------------------------------------------------------


def test_switching_runs_does_not_refetch_or_reverify(env):
    manual = _make_dataset(env.tmp / "manual_celeba", b"\x01", n=300)
    w = _worker(env, data_path=manual)
    w._execute_task(_task("a1", "run_cifar"))
    w._execute_task(_task("b1", "run_celeba"))
    info_calls, manifest_calls = list(env.backend.info_calls), list(env.backend.manifest_calls)
    env.store.downloads.clear()

    with patch("src.worker.cli.dsm.verify_dir", wraps=dsm.verify_dir) as verify:
        for i in range(3):
            assert w._execute_task(_task(f"a{i + 2}", "run_cifar")).success
            assert _input_path(w) == env.tmp / "wcache" / "datasets" / "cifar10" / "clean"
            assert w._execute_task(_task(f"b{i + 2}", "run_celeba")).success
            assert _input_path(w) == manual
    verify.assert_not_called()
    assert env.backend.info_calls == info_calls
    assert env.backend.manifest_calls == manifest_calls
    assert env.store.downloads == []


def test_new_run_on_known_dataset_skips_verification(env):
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x00")
    w = _worker(env, data_root=root)
    w._execute_task(_task("a1", "run_cifar"))

    env.backend.infos["run_cifar_2"] = dict(env.backend.infos["run_cifar"], run_id="run_cifar_2")
    env.backend.manifests["run_cifar_2"] = env.backend.manifests["run_cifar"]
    with patch("src.worker.cli.dsm.verify_dir", wraps=dsm.verify_dir) as verify:
        assert w._execute_task(_task("c1", "run_cifar_2")).success
    verify.assert_not_called()
    assert "run_cifar_2" not in env.backend.manifest_calls
    assert _input_path(w) == root / "cifar10" / "clean"


# ----------------------------------------------------------------------------
# Cache keys and model scripts
# ----------------------------------------------------------------------------


def test_cache_keys_differ_across_datasets_and_are_stable_within_one(env):
    w = _worker(env)
    w._execute_task(_task("a1", "run_cifar"))
    w._execute_task(_task("b1", "run_celeba"))
    w._execute_task(_task("a2", "run_cifar"))
    assert _cache_key(w, "a1") != _cache_key(w, "b1")
    assert _cache_key(w, "a1") == _cache_key(w, "a2")

    # Another run on the same dataset (fresh worker, other host layout) shares the key.
    env.backend.infos["run_cifar_2"] = dict(env.backend.infos["run_cifar"], run_id="run_cifar_2")
    root = env.tmp / "root"
    _make_dataset(root / "cifar10" / "clean", b"\x00")
    w2 = _worker(env, data_root=root)
    w2._execute_task(_task("c1", "run_cifar_2"))
    assert _cache_key(w2, "c1") == _cache_key(w, "a1")


def test_model_script_is_per_run_and_in_the_cache_key(env):
    script_a = env.tmp / "scripts" / "a" / "config_model.py"
    script_b = env.tmp / "scripts" / "b" / "config_model.py"
    script_a.parent.mkdir(parents=True)
    script_b.parent.mkdir(parents=True)
    script_a.write_text("NET = 'resnet20'\n")
    script_b.write_text("NET = 'resnet18'\n")
    env.backend.add_run("run_cifar", "cifar10", env.cifar_src, env.store, model_script=script_a)
    env.backend.add_run("run_cifar_b", "cifar10", env.cifar_src, env.store, model_script=script_b)

    w = _worker(env)
    w._execute_task(_task("a1", "run_cifar"))
    assert w._runner.run_task.call_args.kwargs["model_script_path"] == script_a
    w._execute_task(_task("b1", "run_cifar_b"))
    assert w._runner.run_task.call_args.kwargs["model_script_path"] == script_b
    w._execute_task(_task("a2", "run_cifar"))
    assert w._runner.run_task.call_args.kwargs["model_script_path"] == script_a
    assert _cache_key(w, "a1") != _cache_key(w, "b1")
    assert _cache_key(w, "a1") == _cache_key(w, "a2")


def test_model_script_downloaded_by_hash_when_local_copy_differs(env):
    script = env.tmp / "scripts" / "config_model.py"
    script.parent.mkdir(parents=True)
    script.write_text("NET = 'resnet20'\n")
    env.backend.add_run("run_cifar", "cifar10", env.cifar_src, env.store, model_script=script)
    sha = env.backend.infos["run_cifar"]["model_script_sha256"]
    # The worker's file at the backend's path has other content (another host/checkout).
    script.write_text("NET = 'stale'\n")

    w = _worker(env)
    w._execute_task(_task("a1", "run_cifar"))
    used = w._runner.run_task.call_args.kwargs["model_script_path"]
    assert used == env.tmp / "wcache" / "model_scripts" / sha[:16] / "config_model.py"
    assert used.read_text() == "NET = 'resnet20'\n"


# ----------------------------------------------------------------------------
# verify_dir performance
# ----------------------------------------------------------------------------


def _many_files(d: Path, n: int) -> dict:
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"f{i:03d}.bin").write_bytes(b"x" * 4)
    return dsm.build_manifest(d)


def test_verify_dir_hashes_size_mismatches_in_one_call(tmp_path):
    manifest = _many_files(tmp_path / "d", 50)
    for i in range(40):
        (tmp_path / "d" / f"f{i:03d}.bin").write_bytes(b"y" * 5)

    with patch("src.common.dataset_manifest.hash_files", wraps=dsm.hash_files) as hf, \
         patch("src.common.dataset_manifest._read_sidecar", wraps=dsm._read_sidecar) as rs:
        res = dsm.verify_dir(tmp_path / "d", manifest)
    assert hf.call_count == 1
    assert rs.call_count == 1
    assert len(res.size_mismatch) == 40 and not res.hash_mismatch
    assert res.content_id and res.content_id != manifest["content_id"]
    assert res.content_id == dsm.build_manifest(tmp_path / "d")["content_id"]


def test_verify_dir_quick_stops_without_hashing(tmp_path):
    manifest = _many_files(tmp_path / "d", 50)
    (tmp_path / "d" / "f000.bin").write_bytes(b"y" * 5)

    with patch("src.common.dataset_manifest.hash_files") as hf:
        res = dsm.verify_dir(tmp_path / "d", manifest, quick=True)
    hf.assert_not_called()
    assert not res.ok and res.size_mismatch == ["f000.bin"] and res.content_id is None

    ok = dsm.verify_dir(tmp_path / "d" , dsm.build_manifest(tmp_path / "d"), quick=True)
    assert ok.ok
