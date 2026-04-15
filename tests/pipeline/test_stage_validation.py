"""Tests for pipeline tool stage validation (YAML + image labels, no id-prefix inference)."""

import pytest

from src.pipeline.config_loader import PipelineConfig, StageConfig, DatasetConfig, ModelConfig
from src.pipeline.stage_validation import (
    load_tools_and_validate_pipeline_stages,
    validate_pipeline_tool_stages,
)
from src.pipeline.tools import load_tools_from_yaml


def test_mismatch_defense_stage_raises(tmp_path: Path):
    tools_path = tmp_path / "tools.yaml"
    tools_path.write_text(
        """
tools:
  wrong_place:
    name: wrong
    defense_stage: post_training
    is_baseline: false
    container:
      image: ghcr.io/landseer-project/post_noop_new:v1
      command: python main.py
"""
    )
    pipeline = {
        "pre_training": StageConfig(tools=["wrong_place"]),
        "during_training": StageConfig(tools=[]),
        "post_training": StageConfig(tools=[]),
        "deployment": StageConfig(tools=[]),
    }
    tools = load_tools_from_yaml(str(tools_path))
    with pytest.raises(ValueError, match="wrong_place"):
        validate_pipeline_tool_stages(
            pipeline,
            tools,
            fetch_remote_labels=False,
        )


def test_synonym_in_yaml_accepted(tmp_path: Path):
    tools_path = tmp_path / "tools.yaml"
    tools_path.write_text(
        """
tools:
  in1:
    name: in1
    defense_stage: in
    is_baseline: false
    container:
      image: ghcr.io/landseer-project/in_noop:v7
      command: python main.py
"""
    )
    pipeline = {
        "pre_training": StageConfig(tools=[]),
        "during_training": StageConfig(tools=["in1"]),
        "post_training": StageConfig(tools=[]),
        "deployment": StageConfig(tools=[]),
    }
    tools = load_tools_from_yaml(str(tools_path))
    validate_pipeline_tool_stages(pipeline, tools, fetch_remote_labels=False)


def test_unknown_tool_raises(tmp_path: Path):
    tools_path = tmp_path / "tools.yaml"
    tools_path.write_text(
        """
tools:
  only_tool:
    name: only
    defense_stage: pre_training
    is_baseline: true
    container:
      image: img/x:v1
      command: python main.py
"""
    )
    pipeline = {
        "pre_training": StageConfig(tools=["missing_id"]),
        "during_training": StageConfig(tools=[]),
        "post_training": StageConfig(tools=[]),
        "deployment": StageConfig(tools=[]),
    }
    tools = load_tools_from_yaml(str(tools_path))
    with pytest.raises(ValueError, match="Unknown tool"):
        validate_pipeline_tool_stages(pipeline, tools, fetch_remote_labels=False)


def test_load_tools_and_validate_full_config(tmp_path: Path):
    tools_path = tmp_path / "tools.yaml"
    tools_path.write_text(
        """
tools:
  pre_noop:
    name: noop
    defense_stage: pre_training
    is_baseline: true
    container:
      image: img/pre:v1
      command: python main.py
  in_noop:
    name: in_noop
    defense_stage: during_training
    is_baseline: true
    container:
      image: img/in:v1
      command: python main.py
  post_noop:
    name: post_noop
    defense_stage: post_training
    is_baseline: true
    container:
      image: img/post:v1
      command: python main.py
  deploy_noop:
    name: deploy_noop
    defense_stage: deployment
    is_baseline: true
    container:
      image: img/dep:v1
      command: python main.py
"""
    )
    cfg = PipelineConfig(
        dataset=DatasetConfig(name="cifar10"),
        model=ModelConfig(script="/dev/null"),
        pipeline={
            "pre_training": StageConfig(tools=["pre_noop"]),
            "during_training": StageConfig(tools=["in_noop"]),
            "post_training": StageConfig(tools=["post_noop"]),
            "deployment": StageConfig(tools=["deploy_noop"]),
        },
    )
    load_tools_and_validate_pipeline_stages(
        cfg.pipeline,
        tools_yaml_path=str(tools_path),
        fetch_remote_labels=False,
    )
