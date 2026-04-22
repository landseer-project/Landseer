import yaml
from fastapi.testclient import TestClient

from src.backend.api import app


def test_registry_add_tool_requires_pipeline_key(monkeypatch, tmp_path):
    client = TestClient(app)
    tools_yaml = tmp_path / "tools.yaml"
    monkeypatch.setenv("LANDSEER_PIPELINE_KEYS", "secret-key")
    monkeypatch.setenv("LANDSEER_TOOLS_YAML", str(tools_yaml))

    response = client.post(
        "/registry/tools",
        json={
            "name": "custom pre",
            "image": "ghcr.io/example/custom_pre:v1",
            "command": "python main.py",
            "defense_stage": "pre_training",
        },
    )
    assert response.status_code == 403


def test_registry_add_tool_persists_yaml_with_valid_key(monkeypatch, tmp_path):
    client = TestClient(app)
    tools_yaml = tmp_path / "tools.yaml"
    monkeypatch.setenv("LANDSEER_PIPELINE_KEYS", "secret-key")
    monkeypatch.setenv("LANDSEER_TOOLS_YAML", str(tools_yaml))

    response = client.post(
        "/registry/tools",
        headers={"X-Pipeline-Key": "secret-key"},
        json={
            "name": "custom pre",
            "image": "ghcr.io/example/custom_pre:v1",
            "command": "python main.py",
            "runtime": "docker",
            "is_baseline": False,
            "defense_stage": "pre_training",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["key"] == "custom_pre"
    assert payload["defense_stage"] == "pre_training"

    assert tools_yaml.exists()
    saved = yaml.safe_load(tools_yaml.read_text())
    assert "tools" in saved
    assert "custom_pre" in saved["tools"]
    entry = saved["tools"]["custom_pre"]
    assert entry["name"] == "custom pre"
    assert entry["defense_stage"] == "pre_training"
    assert entry["container"]["image"] == "ghcr.io/example/custom_pre:v1"


def test_start_run_uses_same_pipeline_key_guard(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("LANDSEER_PIPELINE_KEYS", "secret-key")

    blocked = client.post("/api/pipeline-configs/does-not-exist/runs", json={"use_cache": True})
    assert blocked.status_code == 403

    allowed = client.post(
        "/api/pipeline-configs/does-not-exist/runs",
        headers={"X-Pipeline-Key": "secret-key"},
        json={"use_cache": True},
    )
    # Auth passes; endpoint then fails later because config is not present.
    assert allowed.status_code != 403
