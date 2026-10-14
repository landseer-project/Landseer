"""Image label schema: Dockerfile parsing, validation, and legacy-key fallbacks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.pipeline.config_loader import validate_pipeline_tool_dataset_compatibility
from src.pipeline.image_labels import (
    lint_repo,
    parse_dockerfile_labels,
    stage_label,
    supported_datasets_label,
    validate_labels,
)
from src.pipeline.stage_validation import validate_pipeline_tool_stages
from src.pipeline.tools import ContainerConfig, ToolDefinition

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tool_labels(**overrides):
    labels = {
        "org.opencontainers.image.title": "TRADES",
        "org.opencontainers.image.description": "Adversarial training with TRADES",
        "org.opencontainers.image.vendor": "Landseer Project",
        "org.opencontainers.image.source": "https://github.com/landseer-project/trades",
        "org.opencontainers.image.base.name": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        "org.landseer.schema": "1",
        "org.landseer.type": "tool",
        "org.landseer.name": "in_trades",
        "org.landseer.datasets": "cifar10,celeba",
        "org.landseer.modality": "image",
        "org.landseer.framework": "pytorch==2.1.0",
        "org.landseer.python": "3.10",
        "org.landseer.accelerator": "cuda",
        "org.landseer.cuda": "11.8",
        "org.landseer.stage": "during_training",
        "org.landseer.defense_type": "adversarial",
    }
    labels.update(overrides)
    return {k: v for k, v in labels.items() if v is not None}


def test_repo_dockerfiles_follow_label_schema():
    reports = lint_repo(REPO_ROOT)
    assert reports, "no Dockerfiles found under containers/"
    failures = {r.subject: r.errors for r in reports if not r.ok}
    assert not failures, failures


def test_parse_dockerfile_labels_multi_pair_continuation_and_last_stage():
    text = """
FROM python:3.11 AS build
LABEL org.landseer.type="builder"

FROM --platform=linux/amd64 python:3.11-slim
# comment
LABEL a="one two" \\
      # interleaved comment
      b=plain
LABEL legacy value
"""
    labels = parse_dockerfile_labels(text)
    assert labels == {
        "__from__": "python:3.11-slim",
        "a": "one two",
        "b": "plain",
        "legacy": "value",
    }


def test_validate_tool_labels_ok():
    assert validate_labels(_tool_labels()).ok


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"org.landseer.stage": None}, "missing org.landseer.stage"),
        ({"org.landseer.stage": "train"}, "org.landseer.stage='train'"),
        ({"org.landseer.cuda": None}, "requires org.landseer.cuda"),
        ({"org.landseer.datasets": "cifar10, celeba"}, "without spaces"),
        ({"org.landseer.type": "model"}, "org.landseer.type='model'"),
        ({"org.landseer.modality": "image,audio"}, "'audio'"),
        ({"org.landseer.schema": "2"}, "schema"),
    ],
)
def test_validate_tool_labels_errors(overrides, fragment):
    report = validate_labels(_tool_labels(**overrides))
    assert not report.ok
    assert any(fragment in e for e in report.errors), report.errors


def test_built_images_need_build_stamped_labels():
    report = validate_labels(_tool_labels(), built=True)
    assert "missing org.opencontainers.image.revision" in report.errors


def test_expected_type_mismatch():
    report = validate_labels(_tool_labels(), expected_type="evaluator")
    assert any("expected 'evaluator'" in e for e in report.errors)


def test_legacy_keys_warn_and_still_resolve():
    legacy = {
        "org.opencontainers.image.dataset": "CIFAR-10",
        "org.opencontainers.image.stage": "pre_training",
    }
    report = validate_labels(legacy)
    assert any("legacy label org.opencontainers.image.dataset" in w for w in report.warnings)
    assert supported_datasets_label(legacy) == "CIFAR-10"
    assert stage_label(legacy) == "pre_training"

    both = dict(legacy, **{"org.landseer.datasets": "celeba", "org.landseer.stage": "deployment"})
    assert supported_datasets_label(both) == "celeba"
    assert stage_label(both) == "deployment"


def _pipeline(**stages):
    names = ("pre_training", "during_training", "post_training", "deployment")
    return SimpleNamespace(
        pipeline={s: SimpleNamespace(tools=stages.get(s, [])) for s in names}
    )


def _tool(name, image):
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=image, command="python main.py"),
        is_baseline=False,
    )


def test_dataset_compatibility_reads_landseer_datasets_label():
    tools = {"in_fair": _tool("in-fair", "img:fair"), "in_any": _tool("in-any", "img:any")}
    labels = {
        "img:fair": {"org.landseer.datasets": "celeba"},
        "img:any": {"org.landseer.datasets": "*"},
    }
    with patch(
        "src.pipeline.config_loader.get_container_labels_for_image",
        side_effect=lambda image, runtime=None: labels[image],
    ):
        # in_fair also has a hard-coded celeba guardrail; use a different id for the label path.
        tools["in_x"] = _tool("in-x", "img:fair")
        issues = validate_pipeline_tool_dataset_compatibility(
            _pipeline(during_training=["in_x", "in_any"]), tools, "cifar10"
        )
    assert [i["tool_id"] for i in issues] == ["in_x"]


def test_stage_validation_reads_landseer_stage_label():
    tools = {"t": _tool("t", "img:t")}
    with patch(
        "src.pipeline.stage_validation.get_container_labels_for_image",
        return_value={"org.landseer.stage": "post_training"},
    ):
        with pytest.raises(ValueError, match="post_training"):
            validate_pipeline_tool_stages(_pipeline(pre_training=["t"]).pipeline, tools)
