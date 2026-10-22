"""Generic evaluators.yaml options: `input` scope and `enabled_by_attacks`."""

import pytest
from pydantic import ValidationError

from src.pipeline.config_loader import (
    EvaluatorContainerConfig,
    EvaluatorDefinition,
    add_evaluation_tasks_to_workflow,
    load_evaluators_from_yaml,
    select_evaluators_for_attack_config,
)
from src.pipeline.workflow import Workflow


def _defn(name, **kw):
    return EvaluatorDefinition(name=name, container=EvaluatorContainerConfig(image=f"img/{name}:v1", command=""), **kw)


def test_input_defaults_to_final():
    assert _defn("clean").input == "final"


def test_invalid_input_scope_is_rejected():
    with pytest.raises(ValidationError):
        _defn("x", input="whatever")


def test_yaml_options_are_loaded(tmp_path):
    cfg = tmp_path / "evaluators.yaml"
    cfg.write_text(
        "evaluators:\n"
        "  probe:\n"
        "    container: {image: img/probe:v1, command: ''}\n"
        "    input: training_input\n"
        "    enabled_by_attacks: [backdoor]\n"
    )
    probe = load_evaluators_from_yaml(str(cfg))["probe"]
    assert probe.input == "training_input" and probe.enabled_by_attacks == ["backdoor"]


def test_enabled_by_attacks_selects_evaluator():
    evaluators = {
        "clean": _defn("clean"),
        "backdoor": _defn("backdoor"),
        "probe": _defn("probe", enabled_by_attacks=["backdoor", "outlier"]),
        "fairness": _defn("fairness"),
    }
    pick = lambda **a: set(select_evaluators_for_attack_config(evaluators, {"attacks": a}))
    assert pick(backdoor=True) == {"clean", "backdoor", "probe"}
    assert pick(outlier=True) == {"clean", "probe"}
    assert pick(fairness=True) == {"clean", "fairness"}


def test_input_scope_reaches_task_config():
    wf = Workflow(name="wf", pipeline_id="p1")
    tasks = add_evaluation_tasks_to_workflow(
        wf, {"clean": _defn("clean"), "probe": _defn("probe", input="training_input")}, "p1"
    )
    scopes = {t.config["evaluator"]: t.config["input_scope"] for t in tasks}
    assert scopes == {"clean": "final", "probe": "training_input"}


def test_repo_sanitization_entry_uses_generic_options():
    s = load_evaluators_from_yaml("configs/evaluators.yaml")["sanitization"]
    assert s.input == "training_input"
    assert set(s.enabled_by_attacks) == {"backdoor", "outlier"}
    assert s.applies_to["dataset_variants"] == ["poisoned"]
