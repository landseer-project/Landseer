"""
ApptainerRunner Docker parity.

Tests cover:
- --pwd for both exec (command given) and run (image entrypoint)
- --containall, plus --workdir/--overlay on a per-task scratch dir under
  LANDSEER_APPTAINER_SCRATCH that exists during the run and is removed after,
  including on timeout
- container variables passed as APPTAINERENV_* (the host env is dropped)
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.worker.runner import ApptainerRunner


@pytest.fixture
def dirs(tmp_path):
    task_dir = tmp_path / "ws" / "e807ab0e-3f9a2c41d0e7"
    (task_dir / "input").mkdir(parents=True)
    (task_dir / "output").mkdir(parents=True)
    return task_dir / "input", task_dir / "output"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    scratch_root = tmp_path / "node_local"
    scratch_root.mkdir()
    monkeypatch.setenv("LANDSEER_APPTAINER_SCRATCH", str(scratch_root))
    monkeypatch.delenv("LANDSEER_APPTAINER_WORKDIR", raising=False)
    r = ApptainerRunner(workspace_dir=tmp_path / "ws", gpu_id=0)
    r._scratch_root_path = scratch_root
    return r


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def _run(runner, dirs, command="python main.py", side_effect=None):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        seen["existed"] = {
            f: Path(_flag(cmd, f)).is_dir() for f in ("--workdir", "--overlay")
        }
        if side_effect:
            raise side_effect
        return MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("src.worker.runner.subprocess.run", side_effect=fake_run):
        rc, logs, _ = runner.run(
            image="ghcr.io/x/tool:v1",
            command=command,
            input_dir=dirs[0],
            output_dir=dirs[1],
            env={"WORKSPACE": "/", "JUDGE_API_KEY": "k"},
        )
    return rc, logs, seen


def test_exec_uses_docker_parity_flags(runner, dirs):
    rc, _, seen = _run(runner, dirs)
    cmd = seen["cmd"]
    assert rc == 0
    assert cmd[:4] == ["apptainer", "exec", "--pwd", "/app"]
    assert "--containall" in cmd
    assert "--nv" in cmd
    for f in ("--workdir", "--overlay"):
        p = Path(_flag(cmd, f))
        assert seen["existed"][f], f
        assert runner._scratch_root_path in p.parents
        assert "e807ab0e-3f9a2c41d0e7" in str(p)
    assert cmd.index("docker://ghcr.io/x/tool:v1") > cmd.index("--overlay")


def test_entrypoint_run_also_sets_pwd(runner, dirs):
    _, _, seen = _run(runner, dirs, command="")
    assert seen["cmd"][:4] == ["apptainer", "run", "--pwd", "/app"]
    assert "--containall" in seen["cmd"]


def test_container_env_passed_as_apptainerenv(runner, dirs):
    _, _, seen = _run(runner, dirs)
    env = seen["env"]
    assert env["APPTAINERENV_INPUT_DIR"] == "/input"
    assert env["APPTAINERENV_OUTPUT_DIR"] == "/output"
    assert env["APPTAINERENV_WORKSPACE"] == "/"
    assert env["APPTAINERENV_JUDGE_API_KEY"] == "k"
    assert env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "0"


def test_scratch_removed_after_run(runner, dirs):
    _, _, seen = _run(runner, dirs)
    assert not Path(_flag(seen["cmd"], "--overlay")).exists()
    assert list(runner._scratch_root_path.iterdir()) == []


def test_scratch_removed_after_timeout(runner, dirs):
    rc, logs, seen = _run(
        runner, dirs, side_effect=subprocess.TimeoutExpired(cmd="apptainer", timeout=1)
    )
    assert rc == -1 and "Timeout" in logs
    assert list(runner._scratch_root_path.iterdir()) == []
