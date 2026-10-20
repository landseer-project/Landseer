"""
Automatic adapter selection from tool image labels (src/pipeline/tools.py resolve_adapters).

Tests cover:
- Selection from org.landseer.framework / model_input / model_output / model_path
  for each combination (in-training from scratch, post-training on weights,
  deployment without a model output, output-only tools)
- The model path and weights mode handed to the adapters (env, output model_path)
- Explicit tools.yaml adapters and tools.yaml fields override the labels
- When labels are not looked up (standard framework, no adapter catalog, opt-out env)
- Errors: missing model labels, invalid values, no or ambiguous adapters, unsafe paths
- Resolution at task creation: the selected adapters are part of the task hash
- Label lookups are cached per image
- The repository's in_freeat Dockerfile labels select its adapters from configs/tools.yaml
- Label schema validation of the new labels
"""

from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from src.pipeline import tools as tools_mod
from src.pipeline.image_labels import parse_dockerfile_labels, validate_labels
from src.pipeline.tasks import TaskFactory, TaskType, clear_task_registry
from src.pipeline.tools import (
    AdapterDefinition,
    ContainerConfig,
    ToolAdapters,
    ToolDefinition,
    load_tools_from_yaml,
    normalize_framework,
    resolve_adapters,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NS = "org.landseer."

# tests/conftest.py stubs this out for every test; the cache test needs the real one.
_REAL_FETCH = tools_mod._fetch_image_labels

CATALOG_YAML = """\
adapters:
  pytorch_to_tensorflow:
    name: pytorch-to-tensorflow
    source_framework: pytorch
    target_framework: tensorflow
    container:
      image: test/adapter:v1
      command: python main.py --direction input
  config_model_to_tensorflow:
    name: config-model-to-tensorflow
    source_framework: pytorch
    target_framework: tensorflow
    auto_select: false
    container:
      image: test/adapter:v1
      command: python main.py --direction input --weights none
  tensorflow_to_pytorch:
    name: tensorflow-to-pytorch
    source_framework: tensorflow
    target_framework: pytorch
    env:
      EXTRA: "1"
    container:
      image: test/adapter:v1
      command: python main.py --direction output
tools:
  tf_tool:
    name: tf-tool
    defense_stage: during_training
    container:
      image: test/tf_tool:v1
      command: python main.py
  torch_tool:
    name: torch-tool
    defense_stage: during_training
    framework: pytorch
    container:
      image: test/torch_tool:v1
      command: python main.py
"""


@pytest.fixture(autouse=True)
def _clean_task_registry():
    clear_task_registry()
    yield
    clear_task_registry()


class FakeLabels:
    """Label fetcher double: returns fixed labels per image and records lookups."""

    def __init__(self, by_image: Optional[Dict[str, Dict[str, str]]] = None):
        self.by_image = by_image or {}
        self.calls: List[str] = []

    def __call__(self, image: str, runtime: Optional[str] = None) -> Dict[str, str]:
        self.calls.append(image)
        return dict(self.by_image.get(image, {}))


def _labels(framework="tensorflow==2.15.0", model_input="architecture", model_output="model", model_path=None):
    labels = {NS + "framework": framework}
    for key, value in (("model_input", model_input), ("model_output", model_output), ("model_path", model_path)):
        if value is not None:
            labels[NS + key] = value
    return labels


@pytest.fixture
def catalog_tools(tmp_path):
    path = tmp_path / "tools.yaml"
    path.write_text(CATALOG_YAML)
    return load_tools_from_yaml(str(path))


def _resolve(tool: ToolDefinition, labels: Dict[str, str]) -> Dict:
    return resolve_adapters(tool, label_fetcher=FakeLabels({tool.container.image: labels}))


class TestSelection:
    def test_in_training_from_scratch(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels())["adapters"]
        assert spec["input"] == {
            "name": "pytorch-to-tensorflow",
            "image": "test/adapter:v1",
            "command": "python main.py --direction input",
            "env": {"LANDSEER_MODEL_INPUT": "architecture", "LANDSEER_MODEL_PATH": "model_tf/model.keras"},
        }
        assert spec["output"] == {
            "name": "tensorflow-to-pytorch",
            "image": "test/adapter:v1",
            "command": "python main.py --direction output",
            "env": {"EXTRA": "1", "LANDSEER_MODEL_INPUT": "architecture", "LANDSEER_MODEL_PATH": "model_tf/model.keras"},
            "model_path": "model_tf/model.keras",
        }

    def test_post_training_on_weights(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels(model_input="weights"))["adapters"]
        assert spec["input"]["env"]["LANDSEER_MODEL_INPUT"] == "weights"
        assert "output" in spec

    def test_deployment_without_model_output(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels(model_input="weights", model_output="none"))["adapters"]
        assert set(spec) == {"input"}

    def test_output_only(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels(model_input="none"))["adapters"]
        assert set(spec) == {"output"}
        assert spec["output"]["model_path"] == "model_tf/model.keras"

    def test_neither_model_in_nor_out(self, catalog_tools):
        assert _resolve(catalog_tools["tf_tool"], _labels(model_input="none", model_output="none")) == {}

    def test_label_model_path_used(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels(model_path="tf_out/net.keras"))["adapters"]
        assert spec["input"]["env"]["LANDSEER_MODEL_PATH"] == "tf_out/net.keras"
        assert spec["output"]["model_path"] == "tf_out/net.keras"

    def test_framework_aliases(self, catalog_tools):
        for framework in ("tf", "keras==2.15", "TensorFlow", "tensorflow==2.15.0,numpy"):
            assert "adapters" in _resolve(catalog_tools["tf_tool"], _labels(framework=framework))

    def test_auto_select_false_adapter_never_picked(self, catalog_tools):
        spec = _resolve(catalog_tools["tf_tool"], _labels())["adapters"]
        assert spec["input"]["name"] != "config-model-to-tensorflow"


class TestNoAdapters:
    def test_pytorch_label(self, catalog_tools):
        assert _resolve(catalog_tools["tf_tool"], _labels(framework="pytorch==2.1.0")) == {}

    def test_torchvision_label_is_pytorch(self, catalog_tools):
        assert _resolve(catalog_tools["tf_tool"], _labels(framework="torchvision", model_input=None, model_output=None)) == {}

    def test_no_labels(self, catalog_tools):
        assert _resolve(catalog_tools["tf_tool"], {}) == {}

    def test_framework_without_adapters_and_model_labels(self, catalog_tools):
        """numpy / scikit-learn data tools have no model to convert."""
        labels = _labels(framework="scikit-learn==1.3", model_input=None, model_output=None)
        assert _resolve(catalog_tools["tf_tool"], labels) == {}

    def test_yaml_pytorch_framework_skips_label_lookup(self, catalog_tools):
        fetch = FakeLabels()
        assert resolve_adapters(catalog_tools["torch_tool"], label_fetcher=fetch) == {}
        assert fetch.calls == []

    def test_tool_without_catalog_skips_label_lookup(self):
        tool = ToolDefinition(name="t", container=ContainerConfig(image="a:b", command="run"))
        fetch = FakeLabels({"a:b": _labels()})
        assert resolve_adapters(tool, label_fetcher=fetch) == {}
        assert fetch.calls == []

    def test_opt_out_env(self, catalog_tools, monkeypatch):
        monkeypatch.setenv("LANDSEER_AUTO_ADAPTERS", "0")
        fetch = FakeLabels({"test/tf_tool:v1": _labels()})
        assert resolve_adapters(catalog_tools["tf_tool"], label_fetcher=fetch) == {}
        assert fetch.calls == []


class TestOverrides:
    def test_explicit_adapters_win(self, catalog_tools):
        tool = catalog_tools["tf_tool"].model_copy(update={
            "framework": "tensorflow",
            "adapters": ToolAdapters(output=catalog_tools["tf_tool"].adapter_catalog["tensorflow_to_pytorch"]),
        })
        spec = _resolve(tool, _labels())["adapters"]
        # Labels would also add an input adapter; the explicit declaration has only an output one.
        assert set(spec) == {"output"}
        assert spec["output"]["model_path"] == "model_tf/model.keras"

    def test_explicit_adapters_get_label_model_path(self, catalog_tools):
        tool = catalog_tools["tf_tool"].model_copy(update={
            "adapters": ToolAdapters(output=catalog_tools["tf_tool"].adapter_catalog["tensorflow_to_pytorch"]),
        })
        spec = _resolve(tool, _labels(model_path="x/y.keras"))["adapters"]
        assert spec["output"]["model_path"] == "x/y.keras"

    def test_yaml_fields_override_labels(self, tmp_path):
        content = CATALOG_YAML.replace(
            "  tf_tool:\n    name: tf-tool\n",
            "  tf_tool:\n    name: tf-tool\n    model_input: weights\n    model_output: none\n    model_path: m/k.keras\n",
        )
        path = tmp_path / "tools.yaml"
        path.write_text(content)
        tool = load_tools_from_yaml(str(path))["tf_tool"]
        spec = _resolve(tool, _labels())["adapters"]
        assert set(spec) == {"input"}
        assert spec["input"]["env"] == {"LANDSEER_MODEL_INPUT": "weights", "LANDSEER_MODEL_PATH": "m/k.keras"}

    def test_fully_specified_yaml_skips_label_lookup(self, catalog_tools):
        tool = catalog_tools["tf_tool"].model_copy(update={
            "framework": "tensorflow", "model_input": "architecture", "model_output": "model",
            "model_path": "model_tf/model.keras",
        })
        fetch = FakeLabels()
        assert set(resolve_adapters(tool, label_fetcher=fetch)["adapters"]) == {"input", "output"}
        assert fetch.calls == []


class TestErrors:
    def test_tensorflow_without_model_labels(self, catalog_tools):
        with pytest.raises(ValueError, match="model_input"):
            _resolve(catalog_tools["tf_tool"], _labels(model_input=None, model_output=None))

    def test_yaml_framework_with_unreachable_labels(self, catalog_tools):
        tool = catalog_tools["tf_tool"].model_copy(update={"framework": "tensorflow"})
        with pytest.raises(ValueError, match="could not be fetched"):
            _resolve(tool, {})

    def test_invalid_model_input(self, catalog_tools):
        with pytest.raises(ValueError, match="model_input"):
            _resolve(catalog_tools["tf_tool"], _labels(model_input="checkpoint"))

    def test_invalid_model_output(self, catalog_tools):
        with pytest.raises(ValueError, match="model_output"):
            _resolve(catalog_tools["tf_tool"], _labels(model_output="weights"))

    def test_invalid_yaml_value_rejected_at_load(self, tmp_path):
        path = tmp_path / "tools.yaml"
        path.write_text(CATALOG_YAML.replace("  tf_tool:\n    name: tf-tool\n", "  tf_tool:\n    name: tf-tool\n    model_input: maybe\n"))
        with pytest.raises(ValueError, match="model_input"):
            load_tools_from_yaml(str(path))

    @pytest.mark.parametrize("path", ["/output/model.keras", "../model.keras", "a/../../b.keras"])
    def test_unsafe_model_path(self, catalog_tools, path):
        with pytest.raises(ValueError, match="relative"):
            _resolve(catalog_tools["tf_tool"], _labels(model_path=path))

    def test_no_matching_adapter(self, catalog_tools):
        with pytest.raises(ValueError, match="jax -> pytorch|pytorch -> jax"):
            _resolve(catalog_tools["tf_tool"], _labels(framework="jax==0.4"))

    def test_ambiguous_adapters(self, catalog_tools):
        tool = catalog_tools["tf_tool"]
        catalog = dict(tool.adapter_catalog)
        catalog["second"] = catalog["config_model_to_tensorflow"].model_copy(update={"auto_select": True})
        with pytest.raises(ValueError, match="several"):
            _resolve(tool.model_copy(update={"adapter_catalog": catalog}), _labels())


class TestTaskCreation:
    def test_task_factory_resolves_from_labels(self, catalog_tools, monkeypatch):
        fetch = FakeLabels({"test/tf_tool:v1": _labels()})
        monkeypatch.setattr(tools_mod, "_fetch_image_labels", fetch)
        task = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=catalog_tools["tf_tool"], config={"stage": "s"})
        assert set(task.config["adapters"]) == {"input", "output"}
        assert task.config["stage"] == "s"

    def test_label_change_changes_task_hash(self, catalog_tools, monkeypatch):
        tool = catalog_tools["tf_tool"]
        monkeypatch.setattr(tools_mod, "_fetch_image_labels", FakeLabels({"test/tf_tool:v1": _labels()}))
        a = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=tool, config={})
        monkeypatch.setattr(tools_mod, "_fetch_image_labels", FakeLabels({"test/tf_tool:v1": _labels(model_input="weights")}))
        b = TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=tool, config={})
        assert a.get_hash() != b.get_hash()

    def test_resolution_errors_surface_at_task_creation(self, catalog_tools, monkeypatch):
        monkeypatch.setattr(tools_mod, "_fetch_image_labels", FakeLabels({"test/tf_tool:v1": _labels(model_input=None)}))
        with pytest.raises(ValueError, match="model_input"):
            TaskFactory.create_task(task_type=TaskType.IN_TRAINING, tool=catalog_tools["tf_tool"], config={})

    def test_catalog_not_part_of_serialized_tool(self, catalog_tools):
        assert "adapter_catalog" not in catalog_tools["tf_tool"].model_dump()


class TestLabelCache:
    @pytest.fixture(autouse=True)
    def _real_fetch(self, monkeypatch):
        monkeypatch.setattr(tools_mod, "_fetch_image_labels", _REAL_FETCH)
        tools_mod._LABEL_CACHE.clear()
        yield
        tools_mod._LABEL_CACHE.clear()

    def test_lookups_cached_per_image(self):
        with patch("src.pipeline.container_labels.get_container_labels_for_image", return_value={"k": "v"}) as get:
            assert tools_mod._fetch_image_labels("img:1") == {"k": "v"}
            tools_mod._fetch_image_labels("img:1")["k"] = "mutated"
            assert tools_mod._fetch_image_labels("img:1") == {"k": "v"}
            tools_mod._fetch_image_labels("img:2")
        assert get.call_count == 2

    def test_failed_lookups_not_cached(self):
        with patch("src.pipeline.container_labels.get_container_labels_for_image", side_effect=[{}, {"k": "v"}]) as get:
            assert tools_mod._fetch_image_labels("img:1") == {}
            assert tools_mod._fetch_image_labels("img:1") == {"k": "v"}
        assert get.call_count == 2


class TestNormalizeFramework:
    @pytest.mark.parametrize("value,expected", [
        ("tensorflow==2.15.0", "tensorflow"),
        ("pytorch==2.1.2,tensorflow==2.15.1", "pytorch"),
        ("torch>=2", "pytorch"),
        ("Keras", "tensorflow"),
        ("", None),
        (None, None),
    ])
    def test_values(self, value, expected):
        assert normalize_framework(value) == expected


class TestRepository:
    def test_in_freeat_labels_select_its_adapters(self):
        tools = load_tools_from_yaml(str(REPO_ROOT / "configs" / "tools.yaml"))
        tool = tools["in_freeat"]
        assert tool.adapters is None
        labels = parse_dockerfile_labels((REPO_ROOT / "containers" / "tools" / "in_freeat" / "Dockerfile").read_text())
        spec = _resolve(tool, labels)["adapters"]
        assert spec["input"]["name"] == "pytorch-to-tensorflow"
        assert spec["input"]["env"]["LANDSEER_MODEL_INPUT"] == "architecture"
        assert spec["output"]["name"] == "tensorflow-to-pytorch"
        assert spec["output"]["model_path"] == "model_tf/model.keras"

    @pytest.mark.parametrize("key", ["in_madry", "in_madry_celeba"])
    def test_in_madry_labels_select_its_adapters(self, key):
        tools = load_tools_from_yaml(str(REPO_ROOT / "configs" / "tools.yaml"))
        tool = tools[key]
        assert tool.adapters is None
        labels = parse_dockerfile_labels((REPO_ROOT / "containers" / "tools" / "in_madry" / "Dockerfile").read_text())
        assert validate_labels(labels).ok
        spec = _resolve(tool, labels)["adapters"]
        assert spec["input"]["name"] == "pytorch-to-tensorflow"
        assert spec["input"]["env"]["LANDSEER_MODEL_INPUT"] == "architecture"
        assert spec["output"]["name"] == "tensorflow-to-pytorch"
        assert spec["output"]["model_path"] == "model_tf/model.keras"

    def test_pytorch_tools_unaffected(self):
        tools = load_tools_from_yaml(str(REPO_ROOT / "configs" / "tools.yaml"))
        assert _resolve(tools["in_trades"], _labels(framework="pytorch==2.1.0", model_input=None, model_output=None)) == {}


def _tool_labels(**overrides):
    labels = {
        "org.opencontainers.image.title": "TF Tool",
        "org.opencontainers.image.description": "A TensorFlow tool",
        "org.opencontainers.image.vendor": "Landseer Project",
        "org.opencontainers.image.source": "https://github.com/landseer-project/landseer",
        "org.opencontainers.image.base.name": "tensorflow/tensorflow:2.15.0",
        NS + "schema": "1",
        NS + "type": "tool",
        NS + "name": "in_tf",
        NS + "datasets": "cifar10",
        NS + "modality": "image",
        NS + "framework": "tensorflow==2.15.0",
        NS + "python": "3.11",
        NS + "accelerator": "cpu",
        NS + "stage": "during_training",
        NS + "defense_type": "adversarial",
        NS + "model_input": "architecture",
        NS + "model_output": "model",
    }
    labels.update(overrides)
    return {k: v for k, v in labels.items() if v is not None}


class TestLabelSchema:
    def test_valid(self):
        assert validate_labels(_tool_labels()).ok

    def test_tensorflow_tool_requires_model_labels(self):
        report = validate_labels(_tool_labels(**{NS + "model_input": None, NS + "model_output": None}))
        assert any(NS + "model_input" in e for e in report.errors)
        assert any(NS + "model_output" in e for e in report.errors)

    def test_other_frameworks_do_not(self):
        labels = _tool_labels(**{NS + "framework": "numpy==1.24.0", NS + "model_input": None, NS + "model_output": None})
        assert validate_labels(labels).ok

    def test_invalid_values(self):
        report = validate_labels(_tool_labels(**{NS + "model_input": "ckpt", NS + "model_output": "weights"}))
        assert len([e for e in report.errors if "not in" in e]) == 2

    def test_absolute_model_path(self):
        report = validate_labels(_tool_labels(**{NS + "model_path": "/output/m.keras"}))
        assert any("model_path" in e for e in report.errors)
