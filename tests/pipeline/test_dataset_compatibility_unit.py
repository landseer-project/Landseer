from types import SimpleNamespace
from unittest.mock import patch

from src.pipeline.config_loader import (
    normalize_dataset_token,
    parse_supported_datasets_label,
    validate_pipeline_tool_dataset_compatibility,
)
from src.pipeline.tools import ContainerConfig, ToolDefinition


def _tool(name: str, image: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=image, command="python main.py"),
        is_baseline=False,
    )


def test_normalize_dataset_token_aliases():
    assert normalize_dataset_token("CIFAR-10") == "cifar10"
    assert normalize_dataset_token(" CelebA ") == "celeba"
    assert normalize_dataset_token("my_dataset-v1") == "mydatasetv1"


def test_parse_supported_datasets_label_multiple():
    parsed = parse_supported_datasets_label("CIFAR-10, CelebA")
    assert parsed == ["cifar10", "celeba"]


def test_validate_pipeline_tool_dataset_compatibility_flags_mismatch():
    pipeline_cfg = SimpleNamespace(
        pipeline={
            "pre_training": SimpleNamespace(tools=["pre_xgbod"]),
            "during_training": SimpleNamespace(tools=["in_fair"]),
            "post_training": SimpleNamespace(tools=[]),
            "deployment": SimpleNamespace(tools=[]),
        }
    )
    tools_by_id = {
        "pre_xgbod": _tool("pre-xgbod", "ghcr.io/landseer-project/pre_xgbod:v2"),
        "in_fair": _tool("in-fair", "ghcr.io/landseer-project/in_fair:v6"),
    }
    label_map = {
        "ghcr.io/landseer-project/pre_xgbod:v2": {"org.opencontainers.image.dataset": "CIFAR-10"},
        "ghcr.io/landseer-project/in_fair:v6": {"org.opencontainers.image.dataset": "CelebA"},
    }
    with patch(
        "src.pipeline.config_loader.get_container_labels_for_image",
        side_effect=lambda image, runtime=None: label_map.get(image, {}),
    ):
        issues = validate_pipeline_tool_dataset_compatibility(
            pipeline_cfg,
            tools_by_id,
            "cifar10",
            fetch_remote_labels=True,
        )
    assert len(issues) == 1
    assert issues[0]["tool_id"] == "in_fair"
    assert issues[0]["requested_dataset"] == "cifar10"


def test_validate_pipeline_tool_dataset_compatibility_allows_when_label_missing():
    pipeline_cfg = SimpleNamespace(
        pipeline={
            "pre_training": SimpleNamespace(tools=["pre_xgbod"]),
            "during_training": SimpleNamespace(tools=[]),
            "post_training": SimpleNamespace(tools=[]),
            "deployment": SimpleNamespace(tools=[]),
        }
    )
    tools_by_id = {
        "pre_xgbod": _tool("pre-xgbod", "ghcr.io/landseer-project/pre_xgbod:v2"),
    }
    with patch("src.pipeline.config_loader.get_container_labels_for_image", return_value={}):
        issues = validate_pipeline_tool_dataset_compatibility(
            pipeline_cfg,
            tools_by_id,
            "celeba",
            fetch_remote_labels=True,
        )
    assert issues == []
