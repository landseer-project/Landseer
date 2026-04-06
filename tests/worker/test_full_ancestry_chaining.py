"""
Tests for full ancestry chaining in the distributed pipeline.

Verifies that a task receives artifacts from ALL upstream stages (not just
its direct parent), using hard links for zero data duplication.
"""

import os
import stat
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.worker.runner import TaskRunner, _link_or_copy
from src.worker.client import TaskInfo


def make_task(task_id: str, dep_ids: list, task_type: str = "post_training") -> TaskInfo:
    return TaskInfo(
        id=task_id,
        tool_name=f"tool_{task_id}",
        tool_image="test/image:latest",
        tool_command="python main.py",
        tool_runtime=None,
        tool_is_baseline=False,
        config={},
        priority=90,
        status="pending",
        task_type=task_type,
        counter=1,
        workflows=[],
        pipeline_id="pipeline_1",
        dependency_ids=dep_ids,
    )


def mock_container_success(runner):
    """Patch container runner so run() returns exit code 0."""
    mock = MagicMock()
    mock.run.return_value = (0, "ok", "docker run ...")
    mock.pull_image.return_value = True
    runner._container_runner = mock
    return mock


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestGrandparentArtifactsReachGrandchild:
    """A 3-stage chain: A → B → C.  C must see A's model.pt even though it
    only declares B as a direct dependency."""

    def test_grandparent_file_present_in_grandchild_input(self, workspace):
        runner = TaskRunner(workspace_dir=workspace)
        mock_container_success(runner)

        # Stage A output: model.pt
        a_out = workspace / "task_a" / "output"
        a_out.mkdir(parents=True)
        (a_out / "model.pt").write_bytes(b"trained-model")

        # Stage B output: fine-tuned weights (does NOT re-produce model.pt)
        b_out = workspace / "task_b" / "output"
        b_out.mkdir(parents=True)
        (b_out / "weights.pth").write_bytes(b"fine-tuned")

        # B's ancestor chain = [a_out]
        # C's ancestor_dirs  = [a_out, b_out]  (built by Worker._execute_task)
        task_c = make_task("task_c", ["task_b"])
        runner.run_task(task_c, ancestor_dirs=[a_out, b_out])

        c_input = workspace / "task_c" / "input"
        assert (c_input / "model.pt").exists(), "grandparent model.pt must reach task_c"
        assert (c_input / "weights.pth").exists(), "parent weights.pth must reach task_c"

    def test_four_stage_chain_all_artifacts_present(self, workspace):
        runner = TaskRunner(workspace_dir=workspace)
        mock_container_success(runner)

        # pre → during → post → deploy
        pre_out = workspace / "pre" / "output"
        pre_out.mkdir(parents=True)
        (pre_out / "preprocessed.pkl").write_bytes(b"pre")

        during_out = workspace / "during" / "output"
        during_out.mkdir(parents=True)
        (during_out / "model.pt").write_bytes(b"trained")

        post_out = workspace / "post" / "output"
        post_out.mkdir(parents=True)
        (post_out / "pruned.pt").write_bytes(b"pruned")

        deploy_task = make_task("deploy", ["post"], "deployment")
        # ancestor_dirs reflects full chain: pre → during → post
        runner.run_task(deploy_task, ancestor_dirs=[pre_out, during_out, post_out])

        deploy_input = workspace / "deploy" / "input"
        assert (deploy_input / "preprocessed.pkl").exists()
        assert (deploy_input / "model.pt").exists()
        assert (deploy_input / "pruned.pt").exists()


class TestHardLinksNoDataDuplication:
    """Files should be hard-linked (same st_ino), not copied, when on the
    same filesystem."""

    def test_hard_link_same_inode(self, workspace):
        runner = TaskRunner(workspace_dir=workspace)
        mock_container_success(runner)

        src_out = workspace / "src_task" / "output"
        src_out.mkdir(parents=True)
        model = src_out / "model.pt"
        model.write_bytes(b"model-data")

        task = make_task("dst_task", ["src_task"])
        runner.run_task(task, ancestor_dirs=[src_out])

        dst_model = workspace / "dst_task" / "input" / "model.pt"
        assert dst_model.exists()
        assert dst_model.stat().st_ino == model.stat().st_ino, (
            "model.pt should be a hard link (same inode), not a copy"
        )

    def test_link_or_copy_helper_same_fs(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.bin"

        _link_or_copy(src, dst)

        assert dst.exists()
        assert dst.stat().st_ino == src.stat().st_ino


class TestLatestArtifactWins:
    """When both an ancestor and a direct parent produce the same filename,
    the later (direct parent) version must win."""

    def test_direct_parent_model_overrides_grandparent(self, workspace):
        runner = TaskRunner(workspace_dir=workspace)
        mock_container_success(runner)

        # Grandparent: original model
        gp_out = workspace / "grandparent" / "output"
        gp_out.mkdir(parents=True)
        (gp_out / "model.pt").write_bytes(b"original")

        # Parent: updated model
        parent_out = workspace / "parent" / "output"
        parent_out.mkdir(parents=True)
        (parent_out / "model.pt").write_bytes(b"updated")

        task = make_task("child", ["parent"])
        # ancestor_dirs: grandparent first, then parent (later wins)
        runner.run_task(task, ancestor_dirs=[gp_out, parent_out])

        child_model = workspace / "child" / "input" / "model.pt"
        assert child_model.read_bytes() == b"updated", (
            "parent's model.pt must override grandparent's"
        )


class TestWorkerAncestryIntegration:
    """Integration tests for Worker._execute_task ancestry building."""

    def test_ancestry_chain_built_correctly(self, workspace):
        """Worker builds ancestor_dirs by reading _task_outputs of deps."""
        from src.worker.cli import Worker

        worker = Worker(backend_url="http://localhost:8000", workspace_dir=workspace, gpu_id=None)
        worker._init_components()
        worker.use_cache = False

        # Simulate task_a already completed: it has no ancestors
        a_out = workspace / "task_a" / "output"
        a_out.mkdir(parents=True)
        (a_out / "model.pt").write_bytes(b"from-a")
        worker._task_outputs["task_a"] = ("hash_a", a_out, [])

        # task_b depends on task_a
        task_b = make_task("task_b", ["task_a"])

        with patch.object(worker._runner, "run_task") as mock_run:
            from src.worker.runner import ExecutionResult
            b_out = workspace / "task_b" / "output"
            b_out.mkdir(parents=True)
            mock_run.return_value = ExecutionResult(
                success=True, exit_code=0, execution_time_ms=100, output_path=b_out
            )
            worker._execute_task(task_b)

        # Check that run_task was called with ancestor_dirs = [a_out]
        call_kwargs = mock_run.call_args[1]
        assert "ancestor_dirs" in call_kwargs
        assert a_out in call_kwargs["ancestor_dirs"], "a_out must be in task_b's ancestor_dirs"

    def test_cache_hit_propagates_ancestry(self, workspace):
        """A cache-hit task must still record its ancestry so downstream tasks
        can build a correct ancestor_dirs list."""
        from src.worker.cli import Worker

        worker = Worker(backend_url="http://localhost:8000", workspace_dir=workspace, gpu_id=None)
        worker._init_components()
        worker.use_cache = True

        a_out = workspace / "task_a" / "output"
        a_out.mkdir(parents=True)
        (a_out / "model.pt").write_bytes(b"model")
        worker._task_outputs["task_a"] = ("hash_a", a_out, [])

        task_b = make_task("task_b", ["task_a"])

        # Fake a cache hit for task_b
        cached_b = workspace / "cache" / "task_b_out"
        cached_b.mkdir(parents=True)
        (cached_b / "finetuned.pt").write_bytes(b"cached")

        with patch.object(worker, "_check_cache", return_value=cached_b), \
             patch.object(worker, "_compute_cache_key", return_value="hash_b"):
            worker._execute_task(task_b)

        # task_b should be recorded with ancestry [a_out]
        assert "task_b" in worker._task_outputs
        _key, _out, ancestors = worker._task_outputs["task_b"]
        assert a_out in ancestors, "cache-hit task must propagate ancestry"

        # Now simulate task_c depending on task_b
        task_c = make_task("task_c", ["task_b"])
        with patch.object(worker._runner, "run_task") as mock_run:
            from src.worker.runner import ExecutionResult
            c_out = workspace / "task_c" / "output"
            c_out.mkdir(parents=True)
            mock_run.return_value = ExecutionResult(
                success=True, exit_code=0, execution_time_ms=100, output_path=c_out
            )
            worker.use_cache = False
            worker._execute_task(task_c)

        call_kwargs = mock_run.call_args[1]
        ancestor_dirs_c = call_kwargs.get("ancestor_dirs", [])
        # task_c should see both a_out (grandparent) and cached_b (parent)
        assert a_out in ancestor_dirs_c, "grandparent a_out must reach task_c"
        assert cached_b in ancestor_dirs_c, "cached parent b_out must reach task_c"


class TestMergedModelCheckpointIntegrity:
    """Regression: merged model.pt must remain loadable by evaluators (torch.load)."""

    def test_merged_model_pt_loads_as_module(self, workspace):
        runner = TaskRunner(workspace_dir=workspace)
        mock_container_success(runner)

        src_out = workspace / "train" / "output"
        src_out.mkdir(parents=True)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        torch.save(model, src_out / "model.pt")

        task = make_task("eval_task", ["train"], "evaluation")
        runner.run_task(task, ancestor_dirs=[src_out])

        merged = workspace / "eval_task" / "input" / "model.pt"
        assert merged.exists()
        loaded = torch.load(merged, map_location="cpu", weights_only=False)
        assert isinstance(loaded, nn.Module)


class TestRemoteDependencyOutputResolution:
    """When a dependency ran on another worker, _task_outputs may miss it; backend metadata must supply output_path."""

    def test_remote_dep_output_path_in_ancestor_dirs(self, workspace):
        from src.worker.cli import Worker
        from src.worker.runner import ExecutionResult

        remote_out = workspace / "other_worker" / "task_100" / "output"
        remote_out.mkdir(parents=True)
        torch.save(
            nn.Sequential(nn.Flatten(), nn.Linear(3072, 10)),
            remote_out / "model.pt",
        )

        worker = Worker(backend_url="http://localhost:8000", workspace_dir=workspace, gpu_id=None)
        worker._init_components()
        worker.use_cache = False

        def fake_get_task(tid: str):
            if tid != "task_100":
                return None
            return TaskInfo(
                id="task_100",
                tool_name="in_training",
                tool_image="x:latest",
                tool_command="python main.py",
                tool_runtime=None,
                tool_is_baseline=False,
                config={},
                priority=1,
                status="completed",
                task_type="during_training",
                counter=1,
                workflows=[],
                pipeline_id="p1",
                dependency_ids=[],
                cache_key="ck_remote",
                output_path=str(remote_out),
                log_path=None,
            )

        worker._client.get_task = fake_get_task

        eval_task = make_task("eval_200", ["task_100"], "evaluation")
        with patch.object(worker._runner, "run_task") as mock_run:
            e_out = workspace / "eval_200" / "output"
            e_out.mkdir(parents=True)
            mock_run.return_value = ExecutionResult(
                success=True,
                exit_code=0,
                execution_time_ms=10,
                output_path=e_out,
            )
            worker._execute_task(eval_task)

        ancestor_dirs = mock_run.call_args[1]["ancestor_dirs"]
        assert remote_out in ancestor_dirs, "remote dependency output must be merged into input"
