"""
Tool adapter declaration unit tests.

Tests cover:
- Top-level ``adapters:`` section in tools.yaml and per-tool ``adapters: {input, output}``
- Validation: unknown adapter names, unknown positions, malformed sections
- ``framework`` field and the warning for non-standard tools without an output adapter
- adapter_task_config and its injection into task config by TaskFactory
- Task hash/dedup: adapters are part of a task's identity
- The repository's configs/tools.yaml declares in_freeat with its adapters
"""

import logging
from pathlib import Path

import pytest

from src.pipeline.tasks import TaskFactory, TaskType, clear_task_registry, get_or_create_task
from src.pipeline.tools import (
    AdapterDefinition,
    ContainerConfig,
    ToolAdapters,
    ToolDefinition,
    adapter_task_config,
    load_tools_from_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


TOOLS_WITH_ADAPTERS = """\
adapters:
  torch_to_tf:
    name: torch-to-tf
    source_framework: pytorch
    target_framework: tensorflow
    container:
      image: test/adapter:v1
      command: python3 main.py --direction input
    env:
      TRACE_BATCH: 11
  tf_to_torch:
    name: tf-to-torch
    container:
      image: test/adapter:v1
      command: python3 main.py --direction output
tools:
  in_tf_tool:
    name: in-tf-tool
    defense_stage: during_training
    framework: tensorflow
    adapters:
      input: torch_to_tf
      output: tf_to_torch
    container:
      image: test/tf_tool:v1
      command: python3 main.py
  in_torch_tool:
    name: in-torch-tool
    defense_stage: during_training
    container:
      image: test/torch_tool:v1
      command: python3 main.py
"""


@pytest.fixture(autouse=True)
def _clean_task_registry():
    clear_task_registry()
    yield
    clear_task_registry()


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "tools.yaml"
    path.write_text(content)
    return path


def _adapter(name: str, image: str = "test/adapter:v1", command: str = "python3 main.py") -> AdapterDefinition:
    return AdapterDefinition(name=name, container=ContainerConfig(image=image, command=command))


def _tf_tool(output_image: str = "test/adapter:v1") -> ToolDefinition:
    return ToolDefinition(
        name="in-tf-tool",
        framework="tensorflow",
        container=ContainerConfig(image="test/tf_tool:v1", command="python3 main.py"),
        adapters=ToolAdapters(
            input=_adapter("torch-to-tf"),
            output=_adapter("tf-to-torch", image=output_image),
        ),
    )


class TestLoadAdapters:
    def test_adapters_resolved_onto_tool(self, tmp_path):
        tools = load_tools_from_yaml(str(_write(tmp_path, TOOLS_WITH_ADAPTERS)))
        tool = tools["in_tf_tool"]
        assert tool.framework == "tensorflow"
        assert tool.adapters.input.name == "torch-to-tf"
        assert tool.adapters.input.source_framework == "pytorch"
        assert tool.adapters.input.target_framework == "tensorflow"
        assert tool.adapters.input.container.command == "python3 main.py --direction input"
        assert tool.adapters.input.env == {"TRACE_BATCH": "11"}
        assert tool.adapters.output.name == "tf-to-torch"

    def test_tool_without_adapters(self, tmp_path):
        tools = load_tools_from_yaml(str(_write(tmp_path, TOOLS_WITH_ADAPTERS)))
        tool = tools["in_torch_tool"]
        assert tool.adapters is None
        assert tool.framework is None

    def test_yaml_without_adapter_section_still_loads(self, tmp_path):
        content = "tools:\n  t:\n    name: t\n    container:\n      image: a:b\n      command: run\n"
        tools = load_tools_from_yaml(str(_write(tmp_path, content)))
        assert tools["t"].adapters is None

    def test_only_output_adapter(self, tmp_path):
        content = TOOLS_WITH_ADAPTERS.replace("      input: torch_to_tf\n", "")
        tools = load_tools_from_yaml(str(_write(tmp_path, content)))
        assert tools["in_tf_tool"].adapters.input is None
        assert tools["in_tf_tool"].adapters.output.name == "tf-to-torch"

    def test_unknown_adapter_name_rejected(self, tmp_path):
        content = TOOLS_WITH_ADAPTERS.replace("output: tf_to_torch", "output: missing_adapter")
        with pytest.raises(ValueError, match="missing_adapter"):
            load_tools_from_yaml(str(_write(tmp_path, content)))

    def test_unknown_position_rejected(self, tmp_path):
        content = TOOLS_WITH_ADAPTERS.replace("      output: tf_to_torch\n", "      during: tf_to_torch\n")
        with pytest.raises(ValueError, match="during"):
            load_tools_from_yaml(str(_write(tmp_path, content)))

    def test_adapters_section_must_be_mapping(self, tmp_path):
        content = "adapters:\n  - a\ntools:\n  t:\n    name: t\n    container:\n      image: a:b\n      command: run\n"
        with pytest.raises(ValueError, match="mapping"):
            load_tools_from_yaml(str(_write(tmp_path, content)))

    def test_adapter_without_container_rejected(self, tmp_path):
        content = TOOLS_WITH_ADAPTERS.replace(
            "    container:\n      image: test/adapter:v1\n      command: python3 main.py --direction output\n", ""
        )
        with pytest.raises(KeyError):
            load_tools_from_yaml(str(_write(tmp_path, content)))

    def test_warns_for_non_standard_framework_without_output_adapter(self, tmp_path, caplog):
        content = TOOLS_WITH_ADAPTERS.replace("      output: tf_to_torch\n", "")
        with caplog.at_level(logging.WARNING, logger="src.pipeline.tools"):
            load_tools_from_yaml(str(_write(tmp_path, content)))
        assert any("no output adapter" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_output_adapter_present(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="src.pipeline.tools"):
            load_tools_from_yaml(str(_write(tmp_path, TOOLS_WITH_ADAPTERS)))
        assert not any("no output adapter" in r.getMessage() for r in caplog.records)


class TestAdapterTaskConfig:
    def test_empty_for_tool_without_adapters(self):
        tool = ToolDefinition(name="t", container=ContainerConfig(image="a:b", command="run"))
        assert adapter_task_config(tool) == {}

    def test_empty_for_empty_adapters(self):
        tool = ToolDefinition(name="t", container=ContainerConfig(image="a:b", command="run"), adapters=ToolAdapters())
        assert adapter_task_config(tool) == {}

    def test_spec_contents(self):
        assert adapter_task_config(_tf_tool()) == {
            "adapters": {
                "input": {"name": "torch-to-tf", "image": "test/adapter:v1", "command": "python3 main.py", "env": {}},
                "output": {"name": "tf-to-torch", "image": "test/adapter:v1", "command": "python3 main.py", "env": {}},
            }
        }

    def test_model_copy_keeps_adapters(self):
        """config_loader's per-entry command override copies the tool; adapters must survive."""
        tool = _tf_tool()
        copied = tool.model_copy(update={"container": tool.container.model_copy(update={"command": "x"})})
        assert adapter_task_config(copied) == adapter_task_config(tool)


class TestTaskIdentity:
    def test_task_factory_injects_adapters(self):
        task = TaskFactory.create_task(
            task_type=TaskType.IN_TRAINING,
            tool=_tf_tool(),
            config={"stage": "during_training", "tool_name": "in-tf-tool"},
        )
        assert task.config["stage"] == "during_training"
        assert task.config["adapters"]["input"]["name"] == "torch-to-tf"
        assert task.config["adapters"]["output"]["name"] == "tf-to-torch"

    def test_task_factory_leaves_plain_config_untouched(self):
        tool = ToolDefinition(name="t", container=ContainerConfig(image="a:b", command="run"))
        config = {"stage": "during_training"}
        task = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=tool, config=config)
        assert task.config is config
        assert "adapters" not in task.config

    def test_caller_config_not_mutated(self):
        config = {"stage": "during_training"}
        TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=_tf_tool(), config=config)
        assert config == {"stage": "during_training"}

    def test_adapter_change_changes_hash(self):
        a = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=_tf_tool(), config={})
        b = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=_tf_tool(output_image="test/adapter:v2"), config={})
        assert a.get_hash() != b.get_hash()

    def test_adapters_vs_no_adapters_hash_differs(self):
        with_adapters = _tf_tool()
        without = with_adapters.model_copy(update={"adapters": None})
        a = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=with_adapters, config={})
        b = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=without, config={})
        assert a.get_hash() != b.get_hash()

    def test_get_or_create_task_dedups_with_adapters(self):
        a = get_or_create_task(TaskType.IN_TRAINING, _tf_tool(), config={"stage": "during_training"}, pipeline_id="p")
        b = get_or_create_task(TaskType.IN_TRAINING, _tf_tool(), config={"stage": "during_training"}, pipeline_id="p")
        assert a is b

    def test_recreating_from_stored_config_is_stable(self):
        """Rebuilding a task from its persisted config (adapters included) keeps the hash."""
        a = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=_tf_tool(), config={"stage": "s"})
        b = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=_tf_tool(), config=dict(a.config))
        assert a.get_hash() == b.get_hash()


class TestRepositoryToolsYaml:
    def test_in_freeat_declares_adapters(self):
        tools = load_tools_from_yaml(str(REPO_ROOT / "configs" / "tools.yaml"))
        tool = tools["in_freeat"]
        assert tool.defense_stage == "during_training"
        assert tool.framework == "tensorflow"
        assert tool.adapters.input.target_framework == "tensorflow"
        assert tool.adapters.output.source_framework == "tensorflow"
        assert tool.adapters.output.target_framework == "pytorch"

    def test_existing_tools_have_no_adapters(self):
        tools = load_tools_from_yaml(str(REPO_ROOT / "configs" / "tools.yaml"))
        assert tools["in_trades"].adapters is None
        assert adapter_task_config(tools["in_trades"]) == {}
