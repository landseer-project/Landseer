"""
Config loader unit tests.

Tests cover:
- DatasetConfig: fields and defaults
- ModelConfig: script field, framework default
- StageConfig: tools list
- PipelineConfig: validates required stages present; raises for missing stages
- load_pipeline_config: FileNotFoundError, invalid YAML, valid YAML
- EvaluatorDefinition: fields, to_tool() conversion
- _builtin_evaluator_definitions: count and known keys
- init_evaluator_registry: uses built-ins when YAML missing, merges when present
- add_evaluation_tasks_to_workflow: correct number of eval tasks, dependency setup
- make_combinations: total count matches expected Cartesian product
- create_workflow_from_combination: stage priorities, within-stage chaining
"""

import pytest
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock

from src.pipeline.tools import ContainerConfig, ToolDefinition
from src.pipeline.tasks import (
    TaskType,
    TaskStatus,
    EvaluationTask,
    TaskFactory,
    clear_task_registry,
)
from src.pipeline.workflow import Workflow
import src.pipeline.tasks as tasks_module
import src.pipeline.config_loader as config_loader_module
from src.pipeline.config_loader import (
    DatasetConfig,
    ModelConfig,
    StageConfig,
    StageToolConfig,
    PipelineConfig,
    EvaluatorDefinition,
    EvaluatorContainerConfig,
    load_pipeline_config,
    load_evaluators_from_yaml,
    init_evaluator_registry,
    _builtin_evaluator_definitions,
    add_evaluation_tasks_to_workflow,
    make_combinations,
    create_workflow_from_combination,
    get_stage_tool_definitions,
)
import src.pipeline.tools as tools_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    clear_task_registry()
    tasks_module._task_id_counter = 0
    tasks_module._workflow_id_counter = 0
    tasks_module._pipeline_id_counter = 0
    # Clear evaluator registry
    original_eval = config_loader_module._EVALUATOR_REGISTRY.copy()
    original_tools = tools_module._TOOL_REGISTRY.copy()
    yield
    clear_task_registry()
    config_loader_module._EVALUATOR_REGISTRY.clear()
    config_loader_module._EVALUATOR_REGISTRY.update(original_eval)
    tools_module._TOOL_REGISTRY.clear()
    tools_module._TOOL_REGISTRY.update(original_tools)


@pytest.fixture
def minimal_pipeline_yaml(tmp_path: Path) -> Path:
    """A minimal valid pipeline YAML with all required stages."""
    # Use a real model script path we can patch away
    content = """\
dataset:
  name: cifar10
  variant: clean

model:
  script: /dev/null
  framework: pytorch

pipeline:
  pre_training:
    tools: [pre_noop]
  during_training:
    tools: [in_noop]
  post_training:
    tools: [post_noop]
  deployment:
    tools: [deploy_noop]
"""
    f = tmp_path / "pipeline.yaml"
    f.write_text(content)
    return f


@pytest.fixture
def minimal_tools_yaml(tmp_path: Path) -> Path:
    content = """\
tools:
  pre_noop:
    name: noop
    defense_stage: pre_training
    is_baseline: true
    container:
      image: img/pre_noop:v1
      command: python main.py
  in_noop:
    name: in_noop
    defense_stage: during_training
    is_baseline: true
    container:
      image: img/in_noop:v1
      command: python main.py
  post_noop:
    name: post_noop
    defense_stage: post_training
    is_baseline: true
    container:
      image: img/post_noop:v1
      command: python main.py
  deploy_noop:
    name: deploy_noop
    defense_stage: deployment
    is_baseline: true
    container:
      image: img/deploy_noop:v1
      command: python main.py
  tool_b:
    name: tool-b
    defense_stage: pre_training
    is_baseline: false
    container:
      image: img/tool_b:v1
      command: python3 run.py
  tool_b2:
    name: tool-b2
    defense_stage: pre_training
    is_baseline: false
    container:
      image: img/tool_b2:v1
      command: python3 run.py
  in_tool:
    name: in-tool
    defense_stage: during_training
    is_baseline: false
    container:
      image: img/in_tool:v1
      command: python3 run.py
"""
    f = tmp_path / "tools.yaml"
    f.write_text(content)
    return f


def make_tool(
    name: str,
    is_baseline: bool = False,
    defense_stage: str | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        container=ContainerConfig(image=f"img/{name}:v1", command="run"),
        is_baseline=is_baseline,
        defense_stage=defense_stage,
    )


# ============================================================================
# Tests: DatasetConfig
# ============================================================================


class TestDatasetConfig:
    """Tests for DatasetConfig Pydantic model."""

    def test_required_fields(self):
        cfg = DatasetConfig(name="cifar10")
        assert cfg.name == "cifar10"
        assert cfg.variant == "clean"
        assert cfg.params == {}

    def test_custom_variant(self):
        cfg = DatasetConfig(name="mnist", variant="poisoned")
        assert cfg.variant == "poisoned"

    def test_custom_params(self):
        cfg = DatasetConfig(name="d", params={"num_classes": 10})
        assert cfg.params["num_classes"] == 10

    def test_missing_name_raises(self):
        with pytest.raises(Exception):
            DatasetConfig()  # type: ignore[call-arg]


# ============================================================================
# Tests: ModelConfig
# ============================================================================


class TestModelConfig:
    """Tests for ModelConfig Pydantic model."""

    def test_default_framework_pytorch(self, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("# model")
        cfg = ModelConfig(script=str(script))
        assert cfg.framework == "pytorch"

    def test_script_resolved_to_absolute(self, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("# model")
        cfg = ModelConfig(script=str(script))
        assert Path(cfg.script).is_absolute()

    def test_missing_script_does_not_raise_just_warns(self, caplog):
        # ModelConfig only warns, does not raise, for missing scripts
        import logging
        with caplog.at_level(logging.WARNING):
            cfg = ModelConfig(script="/definitely/does/not/exist/model.py")
        assert cfg.script.endswith("model.py")


# ============================================================================
# Tests: StageConfig
# ============================================================================


class TestStageConfig:
    """Tests for StageConfig Pydantic model."""

    def test_empty_tools_default(self):
        cfg = StageConfig()
        assert cfg.tools == []

    def test_tools_list(self):
        cfg = StageConfig(tools=["pre_noop", "pre_xgbod"])
        assert cfg.tools[0].tool == "pre_noop"
        assert len(cfg.tools) == 2

    def test_tools_dict_format_with_command(self):
        cfg = StageConfig(
            tools=[
                {
                    "tool": "pre_watermarkbn",
                    "command": "python main.py --trigger_label 1 --exclude_target_class --trigger_size 12",
                }
            ]
        )
        assert isinstance(cfg.tools[0], StageToolConfig)
        assert cfg.tools[0].tool == "pre_watermarkbn"
        assert "--trigger_label 1" in (cfg.tools[0].command or "")


# ============================================================================
# Tests: PipelineConfig validation
# ============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig validator."""

    def _make_model(self) -> ModelConfig:
        return ModelConfig(script="/dev/null")

    def _make_stage(self, tools=None) -> StageConfig:
        return StageConfig(tools=tools or [])

    def test_valid_pipeline_config(self):
        cfg = PipelineConfig(
            dataset=DatasetConfig(name="cifar10"),
            model=self._make_model(),
            pipeline={
                "pre_training": self._make_stage(),
                "during_training": self._make_stage(),
                "post_training": self._make_stage(),
                "deployment": self._make_stage(),
            }
        )
        assert cfg.dataset.name == "cifar10"

    def test_missing_pre_training_raises(self):
        with pytest.raises(Exception, match="pre_training"):
            PipelineConfig(
                dataset=DatasetConfig(name="d"),
                model=self._make_model(),
                pipeline={
                    "during_training": self._make_stage(),
                    "post_training": self._make_stage(),
                    "deployment": self._make_stage(),
                }
            )

    def test_missing_during_training_raises(self):
        with pytest.raises(Exception, match="during_training"):
            PipelineConfig(
                dataset=DatasetConfig(name="d"),
                model=self._make_model(),
                pipeline={
                    "pre_training": self._make_stage(),
                    "post_training": self._make_stage(),
                    "deployment": self._make_stage(),
                }
            )

    def test_missing_multiple_stages_raises(self):
        with pytest.raises(Exception):
            PipelineConfig(
                dataset=DatasetConfig(name="d"),
                model=self._make_model(),
                pipeline={}
            )


# ============================================================================
# Tests: load_pipeline_config
# ============================================================================


class TestLoadPipelineConfig:
    """Tests for load_pipeline_config."""

    def test_loads_valid_config(self, minimal_pipeline_yaml):
        cfg = load_pipeline_config(
            str(minimal_pipeline_yaml),
            fetch_remote_labels_for_stage_validation=False,
        )
        assert isinstance(cfg, PipelineConfig)
        assert cfg.dataset.name == "cifar10"

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent/pipeline.yaml")

    def test_raises_for_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  - bad: [missing")
        with pytest.raises(ValueError):
            load_pipeline_config(str(bad))

    def test_raises_for_missing_required_stage(self, tmp_path):
        content = """\
dataset:
  name: d
model:
  script: /dev/null
pipeline:
  pre_training:
    tools: []
"""
        f = tmp_path / "p.yaml"
        f.write_text(content)
        with pytest.raises(ValueError):
            load_pipeline_config(str(f))

    def test_stages_loaded_correctly(self, minimal_pipeline_yaml):
        cfg = load_pipeline_config(
            str(minimal_pipeline_yaml),
            fetch_remote_labels_for_stage_validation=False,
        )
        assert "pre_training" in cfg.pipeline
        assert "during_training" in cfg.pipeline
        assert "post_training" in cfg.pipeline
        assert "deployment" in cfg.pipeline


# ============================================================================
# Tests: EvaluatorDefinition
# ============================================================================


class TestEvaluatorDefinition:
    """Tests for EvaluatorDefinition model and to_tool() conversion."""

    def test_basic_construction(self):
        defn = EvaluatorDefinition(
            name="clean",
            container=EvaluatorContainerConfig(
                image="img/clean:v1", command=""
            ),
            required_artifacts=[],
            metrics=["clean_accuracy"],
        )
        assert defn.name == "clean"
        assert defn.metrics == ["clean_accuracy"]

    def test_to_tool_returns_tool_definition(self):
        defn = EvaluatorDefinition(
            name="adversarial",
            container=EvaluatorContainerConfig(image="img/adv:v1", command="run"),
        )
        tool = defn.to_tool()
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "adversarial"
        assert tool.is_baseline is False

    def test_to_tool_image_propagated(self):
        defn = EvaluatorDefinition(
            name="ood",
            container=EvaluatorContainerConfig(image="img/ood:v2", command=""),
        )
        tool = defn.to_tool()
        assert tool.container.image == "img/ood:v2"

    def test_to_tool_runtime_propagated(self):
        defn = EvaluatorDefinition(
            name="e",
            container=EvaluatorContainerConfig(image="img:v1", command="", runtime="docker"),
        )
        tool = defn.to_tool()
        assert tool.container.runtime == "docker"

    def test_default_required_artifacts_empty(self):
        defn = EvaluatorDefinition(
            name="e",
            container=EvaluatorContainerConfig(image="i", command="c")
        )
        assert defn.required_artifacts == []


# ============================================================================
# Tests: _builtin_evaluator_definitions
# ============================================================================


class TestBuiltinEvaluatorDefinitions:
    """Tests for _builtin_evaluator_definitions."""

    def test_returns_seven_evaluators(self):
        builtins = _builtin_evaluator_definitions()
        assert len(builtins) == 7

    @pytest.mark.parametrize("key", ["clean", "backdoor", "adversarial", "fairness",
                                      "fingerprinting", "ood", "watermark"])
    def test_all_expected_keys_present(self, key):
        builtins = _builtin_evaluator_definitions()
        assert key in builtins

    def test_clean_evaluator_no_required_artifacts(self):
        builtins = _builtin_evaluator_definitions()
        assert builtins["clean"].required_artifacts == []

    def test_backdoor_requires_poisoning_metadata(self):
        builtins = _builtin_evaluator_definitions()
        assert "poisoning_metadata.json" in builtins["backdoor"].required_artifacts

    def test_watermark_requires_watermark_key(self):
        builtins = _builtin_evaluator_definitions()
        assert "watermark_key.json" in builtins["watermark"].required_artifacts

    def test_fairness_requires_sensitive_attributes(self):
        builtins = _builtin_evaluator_definitions()
        assert "sensitive_attributes.npy" in builtins["fairness"].required_artifacts

    def test_all_evaluators_have_container_image(self):
        builtins = _builtin_evaluator_definitions()
        for name, defn in builtins.items():
            assert defn.container.image, f"Evaluator {name} missing image"


# ============================================================================
# Tests: init_evaluator_registry
# ============================================================================


class TestInitEvaluatorRegistry:
    """Tests for init_evaluator_registry."""

    def test_uses_builtins_when_yaml_missing(self):
        init_evaluator_registry("/nonexistent/evaluators.yaml")
        assert len(config_loader_module._EVALUATOR_REGISTRY) == 7

    def test_merges_custom_with_builtins(self, tmp_path):
        custom_yaml = tmp_path / "evaluators.yaml"
        custom_yaml.write_text("""\
evaluators:
  custom_eval:
    name: custom-eval
    container:
      image: img/custom:v1
      command: run
    metrics: [my_metric]
""")
        init_evaluator_registry(str(custom_yaml))
        # Built-ins (7) + custom (1) = 8
        assert len(config_loader_module._EVALUATOR_REGISTRY) == 8
        assert "custom_eval" in config_loader_module._EVALUATOR_REGISTRY

    def test_builtin_can_be_overridden(self, tmp_path):
        custom_yaml = tmp_path / "evaluators.yaml"
        custom_yaml.write_text("""\
evaluators:
  clean:
    name: clean-override
    container:
      image: img/clean-custom:v99
      command: run-custom
    metrics: [custom_accuracy]
""")
        init_evaluator_registry(str(custom_yaml))
        clean = config_loader_module._EVALUATOR_REGISTRY["clean"]
        assert clean.container.image == "img/clean-custom:v99"


# ============================================================================
# Tests: add_evaluation_tasks_to_workflow
# ============================================================================


class TestAddEvaluationTasksToWorkflow:
    """Tests for add_evaluation_tasks_to_workflow."""

    def test_adds_one_task_per_evaluator(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        evaluators = _builtin_evaluator_definitions()

        tasks = add_evaluation_tasks_to_workflow(
            workflow=wf, evaluators=evaluators, pipeline_id="p1"
        )

        assert len(tasks) == len(evaluators)
        assert len(wf.tasks) == len(evaluators)

    def test_all_added_tasks_are_evaluation_type(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        evaluators = _builtin_evaluator_definitions()
        tasks = add_evaluation_tasks_to_workflow(wf, evaluators, "p1")

        for task in tasks:
            assert isinstance(task, EvaluationTask)

    def test_all_eval_tasks_have_low_priority(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        evaluators = _builtin_evaluator_definitions()
        tasks = add_evaluation_tasks_to_workflow(wf, evaluators, "p1")

        for task in tasks:
            assert task.priority == 50

    def test_eval_tasks_depend_on_last_deployment_task(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        deploy_tool = make_tool("deploy-tool")
        last_deploy = TaskFactory.create_task(TaskType.DEPLOYMENT, tool=deploy_tool)

        evaluators = {"clean": _builtin_evaluator_definitions()["clean"]}
        tasks = add_evaluation_tasks_to_workflow(wf, evaluators, "p1", last_deploy)

        assert last_deploy in tasks[0].dependencies

    def test_eval_tasks_have_no_dep_when_no_deployment_task(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        evaluators = {"clean": _builtin_evaluator_definitions()["clean"]}
        tasks = add_evaluation_tasks_to_workflow(wf, evaluators, "p1", last_deployment_task=None)

        assert tasks[0].dependencies == []

    def test_required_artifacts_propagated(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        evaluators = {"backdoor": _builtin_evaluator_definitions()["backdoor"]}
        tasks = add_evaluation_tasks_to_workflow(wf, evaluators, "p1")

        assert "poisoning_metadata.json" in tasks[0].required_artifacts

    def test_empty_evaluators_adds_no_tasks(self):
        wf = Workflow(name="wf", pipeline_id="p1")
        tasks = add_evaluation_tasks_to_workflow(wf, {}, "p1")
        assert tasks == []
        assert wf.tasks == []


# ============================================================================
# Tests: make_combinations
# ============================================================================


class TestMakeCombinations:
    """Tests for make_combinations Cartesian product logic."""

    def _build_config(
        self, tmp_path: Path,
        pre_tools: list, in_tools: list,
        post_tools: list, deploy_tools: list,
        tools_yaml: Path
    ) -> PipelineConfig:
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(tools_yaml))
        return PipelineConfig(
            dataset=DatasetConfig(name="cifar10"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=pre_tools),
                "during_training": StageConfig(tools=in_tools),
                "post_training": StageConfig(tools=post_tools),
                "deployment": StageConfig(tools=deploy_tools),
            }
        )

    def test_all_baselines_single_combination(self, minimal_tools_yaml):
        # Each stage has only 1 baseline tool → 1 option per stage → 1 combination
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(minimal_tools_yaml))
        config = PipelineConfig(
            dataset=DatasetConfig(name="d"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=["pre_noop"]),
                "during_training": StageConfig(tools=["in_noop"]),
                "post_training": StageConfig(tools=["post_noop"]),
                "deployment": StageConfig(tools=["deploy_noop"]),
            }
        )
        combos = make_combinations(config)
        assert len(combos) == 1

    def test_one_actual_pre_tool_increases_combos(self, minimal_tools_yaml):
        # pre has [tool_b, pre_noop]: 2 options (tool_b, noop)
        # during = [in_noop]: 1 option
        # post = [post_noop]: 1 option
        # deploy = [deploy_noop]: 1 option
        # total = 2
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(minimal_tools_yaml))
        config = PipelineConfig(
            dataset=DatasetConfig(name="d"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=["tool_b", "pre_noop"]),
                "during_training": StageConfig(tools=["in_noop"]),
                "post_training": StageConfig(tools=["post_noop"]),
                "deployment": StageConfig(tools=["deploy_noop"]),
            }
        )
        combos = make_combinations(config)
        # pre has [tool_b], [noop] = 2 options (single actual + baseline)
        assert len(combos) == 2

    def test_two_actual_pre_tools_five_combos(self, minimal_tools_yaml):
        # pre: [tool_b, tool_b2, pre_noop] → 5 options
        # other stages: 1 option each
        # total = 5
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(minimal_tools_yaml))
        config = PipelineConfig(
            dataset=DatasetConfig(name="d"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=["tool_b", "tool_b2", "pre_noop"]),
                "during_training": StageConfig(tools=["in_noop"]),
                "post_training": StageConfig(tools=["post_noop"]),
                "deployment": StageConfig(tools=["deploy_noop"]),
            }
        )
        combos = make_combinations(config)
        assert len(combos) == 5

    def test_two_during_training_tools_two_combos(self, minimal_tools_yaml):
        # during: [in_tool, in_noop] → 2 options (single tool constraint)
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(minimal_tools_yaml))
        config = PipelineConfig(
            dataset=DatasetConfig(name="d"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=["pre_noop"]),
                "during_training": StageConfig(tools=["in_tool", "in_noop"]),
                "post_training": StageConfig(tools=["post_noop"]),
                "deployment": StageConfig(tools=["deploy_noop"]),
            }
        )
        combos = make_combinations(config)
        assert len(combos) == 2

    def test_combinations_are_dicts_with_stage_keys(self, minimal_tools_yaml):
        from src.pipeline.tools import init_tool_registry
        init_tool_registry(str(minimal_tools_yaml))
        config = PipelineConfig(
            dataset=DatasetConfig(name="d"),
            model=ModelConfig(script="/dev/null"),
            pipeline={
                "pre_training": StageConfig(tools=["pre_noop"]),
                "during_training": StageConfig(tools=["in_noop"]),
                "post_training": StageConfig(tools=["post_noop"]),
                "deployment": StageConfig(tools=["deploy_noop"]),
            }
        )
        combos = make_combinations(config)
        for combo in combos:
            assert "pre_training" in combo
            assert "during_training" in combo
            assert "post_training" in combo
            assert "deployment" in combo
