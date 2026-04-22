from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.backend.api import app


@contextmanager
def _fake_session_scope():
    yield object()


class _FakeRunRepoEmpty:
    def __init__(self, _session):
        pass

    def get_active_runs_for_config(self, _config_id):
        return []

    def get_next_run_number(self, _config_id):
        return 1

    def create(self, run_data):
        return SimpleNamespace(
            id=run_data["id"],
            pipeline_config_id=run_data["pipeline_config_id"],
            run_number=run_data["run_number"],
            use_cache=run_data["use_cache"],
            status=SimpleNamespace(value="pending"),
            error_message=None,
            created_at=SimpleNamespace(isoformat=lambda: "2026-01-01T00:00:00"),
            started_at=None,
            completed_at=None,
            tools_config=run_data.get("tools_config"),
        )

    def update_status(self, _run_id, _status):
        return None

    def get_by_id(self, _run_id):
        return None


def test_start_run_rejects_incompatible_tool_dataset(tmp_path: Path):
    client = TestClient(app)
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text("dataset:\n  name: cifar10\n  variant: clean\nmodel:\n  script: /tmp/model.py\npipeline: {}\n")
    cfg = SimpleNamespace(
        id="config_test",
        name="test",
        config_path=str(cfg_file),
        attack_config_path=None,
    )
    fake_loaded = SimpleNamespace(
        dataset=SimpleNamespace(name="cifar10"),
        pipeline={"pre_training": SimpleNamespace(tools=[]), "during_training": SimpleNamespace(tools=["in_fair"]), "post_training": SimpleNamespace(tools=[]), "deployment": SimpleNamespace(tools=[])},
    )
    with patch("src.backend.config_discovery.get_config_by_id", return_value=cfg), patch(
        "src.db.session_scope", _fake_session_scope
    ), patch("src.db.PipelineRunRepository", _FakeRunRepoEmpty), patch(
        "src.pipeline.config_loader.load_pipeline_config", return_value=fake_loaded
    ), patch(
        "src.pipeline.stage_validation.load_tools_and_validate_pipeline_stages",
        return_value={},
    ), patch(
        "src.pipeline.config_loader.validate_pipeline_tool_dataset_compatibility",
        return_value=[{
            "stage": "during_training",
            "tool_id": "in_fair",
            "tool_name": "in-fair",
            "image": "ghcr.io/landseer-project/in_fair:v6",
            "requested_dataset": "cifar10",
            "supported_datasets": "celeba",
        }],
    ):
        response = client.post(
            "/api/pipeline-configs/config_test/runs",
            json={"use_cache": True, "dataset_name": "cifar10"},
        )
    assert response.status_code == 400
    assert "compatibility check failed" in response.json()["detail"].lower()
    assert "in_fair" in response.json()["detail"]
