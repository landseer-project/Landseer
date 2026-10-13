"""Unit tests for Landseer control-plane CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.client import LandseerApiClient, LandseerApiError
from src.cli.main import cli


def test_config_id_from_yaml():
    assert (
        LandseerApiClient.config_id_from_yaml("configs/pipeline/trades.yaml")
        == "config_trades"
    )
    assert LandseerApiClient.config_id_from_yaml("/tmp/mini.yml") == "config_mini"
    with pytest.raises(LandseerApiError):
        LandseerApiClient.config_id_from_yaml("trades")


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "health" in result.output
    assert "progress" in result.output
    assert "worker" in result.output


def test_run_start_posts_yaml_path(tmp_path: Path):
    runner = CliRunner()
    yaml_path = tmp_path / "trades.yaml"
    yaml_path.write_text("pipeline: {}\n")

    mock_client = MagicMock()
    mock_client.config_id_from_yaml.return_value = "config_trades"
    mock_client.start_run.return_value = {
        "id": "run_abc",
        "experiment_id": "config_trades",
        "run_number": 1,
        "status": "pending",
        "use_cache": True,
        "created_at": "2026-01-01T00:00:00",
        "started_at": None,
        "error_message": None,
    }

    with patch("src.cli.main.LandseerApiClient", return_value=mock_client):
        result = runner.invoke(cli, ["run", "start", str(yaml_path)])

    assert result.exit_code == 0, result.output
    mock_client.config_id_from_yaml.assert_called_once_with(str(yaml_path))
    mock_client.start_run.assert_called_once_with("config_trades")
    assert "run_abc" in result.output


def test_run_list_and_show():
    runner = CliRunner()
    mock_client = MagicMock()
    mock_client.list_runs.return_value = {
        "runs": [
            {
                "id": "run_1",
                "experiment_id": "config_trades",
                "run_number": 1,
                "status": "running",
                "use_cache": True,
                "created_at": "t0",
            }
        ],
        "total": 1,
    }
    mock_client.get_run.return_value = {
        "id": "run_1",
        "experiment_id": "config_trades",
        "run_number": 1,
        "status": "running",
        "use_cache": True,
        "created_at": "t0",
        "started_at": "t1",
        "completed_at": None,
        "error_message": None,
    }

    with patch("src.cli.main.LandseerApiClient", return_value=mock_client):
        listed = runner.invoke(cli, ["run", "list"])
        shown = runner.invoke(cli, ["run", "show", "run_1"])

    assert listed.exit_code == 0, listed.output
    assert "run_1" in listed.output
    assert shown.exit_code == 0, shown.output
    assert "running" in shown.output


def test_task_list_for_run():
    runner = CliRunner()
    mock_client = MagicMock()
    mock_client.list_tasks.return_value = {
        "tasks": [
            {
                "id": "task_1",
                "tool": {"name": "pre_noop"},
                "status": "completed",
                "task_type": "tool",
                "priority": 1,
            }
        ],
        "total": 1,
    }

    with patch("src.cli.main.LandseerApiClient", return_value=mock_client):
        result = runner.invoke(
            cli, ["task", "list", "run_abc", "--status", "completed"]
        )

    assert result.exit_code == 0, result.output
    mock_client.list_tasks.assert_called_once_with("run_abc", status="completed")
    assert "task_1" in result.output
    assert "pre_noop" in result.output


def test_status_health_progress_workers():
    runner = CliRunner()
    mock_client = MagicMock()
    mock_client.health.return_value = {
        "status": "ok",
        "scheduler_active": True,
        "timestamp": "t",
    }
    mock_client.progress.return_value = {
        "total": 10,
        "pending": 2,
        "running": 1,
        "completed": 7,
        "failed": 0,
        "progress_percent": 70.0,
        "is_complete": False,
    }
    mock_client.list_workers.return_value = {
        "workers": [
            {
                "worker_id": "w1",
                "hostname": "host",
                "status": "idle",
                "current_task_id": None,
                "tasks_completed": 3,
                "tasks_failed": 0,
            }
        ],
        "total": 1,
        "active": 1,
    }

    with patch("src.cli.main.LandseerApiClient", return_value=mock_client):
        health = runner.invoke(cli, ["health"])
        progress = runner.invoke(cli, ["progress", "--run-id", "run_1"])
        status = runner.invoke(cli, ["status"])
        workers = runner.invoke(cli, ["worker", "list"])

    assert health.exit_code == 0
    assert progress.exit_code == 0
    assert status.exit_code == 0
    assert workers.exit_code == 0
    mock_client.progress.assert_any_call(run_id="run_1")
    assert "w1" in workers.output
