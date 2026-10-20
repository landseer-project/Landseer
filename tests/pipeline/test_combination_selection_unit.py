"""Unit tests for the `combinations:` include/exclude block in pipeline configs."""

from pathlib import Path

import pytest

import src.pipeline.tools as tools_module
from src.pipeline.config_loader import (
    CombinationSelection,
    DatasetConfig,
    ModelConfig,
    PipelineConfig,
    StageConfig,
    make_combinations,
    select_combinations,
)
from src.pipeline.tools import init_tool_registry

TOOLS_YAML = """\
tools:
  pre_noop:
    name: noop
    defense_stage: pre_training
    is_baseline: true
    container: {image: img/pre_noop:v1, command: python main.py}
  pre_wm:
    name: pre-wm
    defense_stage: pre_training
    container: {image: img/pre_wm:v1, command: python main.py}
  in_noop:
    name: in-noop
    defense_stage: during_training
    is_baseline: true
    container: {image: img/in_noop:v1, command: python main.py}
  in_fair:
    name: in-fair
    defense_stage: during_training
    container: {image: img/in_fair:v1, command: python main.py}
  post_noop:
    name: post_noop
    defense_stage: post_training
    is_baseline: true
    container: {image: img/post_noop:v1, command: python main.py}
  deploy_noop:
    name: deploy_noop
    defense_stage: deployment
    is_baseline: true
    container: {image: img/deploy_noop:v1, command: python main.py}
  deploy_shap:
    name: deploy-shap
    defense_stage: deployment
    container: {image: img/deploy_shap:v1, command: python main.py}
"""


@pytest.fixture(autouse=True)
def registry(tmp_path: Path):
    original = tools_module._TOOL_REGISTRY.copy()
    tools_yaml = tmp_path / "tools.yaml"
    tools_yaml.write_text(TOOLS_YAML)
    init_tool_registry(str(tools_yaml))
    yield
    tools_module._TOOL_REGISTRY.clear()
    tools_module._TOOL_REGISTRY.update(original)


def _config(combinations=None) -> PipelineConfig:
    # 2 pre x 2 in x 1 post x 2 deploy = 8 combinations, like the CelebA configs
    return PipelineConfig.model_validate({
        "dataset": {"name": "celeba"},
        "model": {"script": "/dev/null"},
        "pipeline": {
            "pre_training": {"tools": ["pre_noop", "pre_wm"]},
            "during_training": {"tools": ["in_noop", "in_fair"]},
            "post_training": {"tools": ["post_noop"]},
            "deployment": {"tools": ["deploy_noop", "deploy_shap"]},
        },
        "combinations": combinations,
    })


def _select(combinations):
    config = _config(combinations)
    return select_combinations(make_combinations(config), config.combinations)


def _stage(combo, stage):
    return [t.name for t in combo[stage]]


def test_no_selection_keeps_all_in_order():
    selected = _select(None)
    assert [cid for cid, _ in selected] == [f"comb_{i:03d}" for i in range(1, 9)]


def test_exclude_by_stage_tool_keeps_original_ids():
    selected = _select({"exclude": [{"during_training": "in_noop"}]})
    assert len(selected) == 4
    assert all(_stage(c, "during_training") == ["in-fair"] for _, c in selected)
    full_ids = {cid for cid, c in _select(None) if _stage(c, "during_training") == ["in-fair"]}
    assert {cid for cid, _ in selected} == full_ids


def test_include_rule_requires_all_stages_to_match():
    selected = _select({"include": [{"during_training": "in_fair", "deployment": "deploy_shap"}]})
    assert len(selected) == 2
    for _, combo in selected:
        assert _stage(combo, "during_training") == ["in-fair"]
        assert _stage(combo, "deployment") == ["deploy-shap"]


def test_bare_list_is_include_and_accepts_ids_and_display_names():
    selected = _select(["comb_001", {"pre_training": "pre-wm", "during_training": "in-fair",
                                     "deployment": "deploy_noop"}])
    assert len(selected) == 2
    assert selected[0][0] == "comb_001"


def test_include_and_exclude_combine():
    selected = _select({"include": [{"during_training": "in_fair"}],
                        "exclude": [{"pre_training": "pre_noop"}]})
    assert len(selected) == 2
    assert all(_stage(c, "pre_training") == ["pre-wm"] for _, c in selected)


@pytest.mark.parametrize("rule", [
    {"during_training": "in_trades"},          # tool not in this config
    {"during_trainin": "in_fair"},             # stage typo
    {"pre_training": "in_fair"},               # tool from another stage
    "comb_099",                                # combo id out of range
])
def test_unknown_rules_raise(rule):
    with pytest.raises(ValueError, match="combinations:"):
        _select({"include": [rule]})


def test_selecting_nothing_raises():
    with pytest.raises(ValueError, match="leave none"):
        _select({"include": [{"during_training": "in_fair"}],
                 "exclude": [{"during_training": "in_fair"}]})


def test_selection_model_accepts_bare_list():
    assert CombinationSelection.model_validate(["comb_002"]).include == ["comb_002"]
