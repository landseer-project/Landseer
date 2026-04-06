"""
Tool registry unit tests.

Tests cover:
- ContainerConfig and ToolDefinition Pydantic validation
- load_tools_from_yaml: happy path, missing file, malformed YAML, missing 'tools' key
- init_tool_registry populates the global registry
- get_tool: found and not-found
- get_all_tools: returns a copy, not a reference
- is_baseline flag propagation
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.tools import (
    ContainerConfig,
    ToolDefinition,
    load_tools_from_yaml,
    init_tool_registry,
    get_tool,
    get_all_tools,
)
import src.pipeline.tools as tools_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset the global tool registry before and after every test."""
    original = tools_module._TOOL_REGISTRY.copy()
    tools_module._TOOL_REGISTRY.clear()
    yield
    tools_module._TOOL_REGISTRY.clear()
    tools_module._TOOL_REGISTRY.update(original)


@pytest.fixture
def minimal_tools_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid tools.yaml to a temp file and return its path."""
    yaml_content = """\
tools:
  pre_noop:
    name: noop
    is_baseline: true
    container:
      image: test/noop:v1
      command: python main.py
  pre_tool:
    name: pre-tool
    is_baseline: false
    container:
      image: test/pre_tool:v2
      command: python3 run.py
"""
    yaml_file = tmp_path / "tools.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def multi_tools_yaml(tmp_path: Path) -> Path:
    """A tools.yaml with tools spanning all stages."""
    yaml_content = """\
tools:
  pre_noop:
    name: noop
    is_baseline: true
    container:
      image: img/pre_noop:v1
      command: python main.py
      runtime: null
  in_dp:
    name: in-dp
    is_baseline: false
    container:
      image: img/in_dp:v1
      command: python3 main.py
  deploy_tool:
    name: deploy-tool
    is_baseline: false
    container:
      image: img/deploy:v1
      command: python3 deploy.py
      runtime: docker
"""
    f = tmp_path / "tools.yaml"
    f.write_text(yaml_content)
    return f


# ============================================================================
# Tests: ContainerConfig
# ============================================================================


class TestContainerConfig:
    """Tests for ContainerConfig Pydantic model."""

    # -- happy path --

    def test_minimal_construction(self):
        # Arrange / Act
        cfg = ContainerConfig(image="repo/img:v1", command="python run.py")

        # Assert
        assert cfg.image == "repo/img:v1"
        assert cfg.command == "python run.py"
        assert cfg.runtime is None

    def test_with_runtime(self):
        # Arrange / Act
        cfg = ContainerConfig(image="img:v1", command="run", runtime="docker")

        # Assert
        assert cfg.runtime == "docker"

    # -- edge cases --

    def test_empty_command_allowed(self):
        # Pydantic should not reject an empty string command
        cfg = ContainerConfig(image="img:v1", command="")
        assert cfg.command == ""

    def test_missing_image_raises(self):
        # Pydantic validation requires 'image'
        with pytest.raises(Exception):
            ContainerConfig(command="python run.py")  # type: ignore[call-arg]

    def test_missing_command_raises(self):
        with pytest.raises(Exception):
            ContainerConfig(image="img:v1")  # type: ignore[call-arg]


# ============================================================================
# Tests: ToolDefinition
# ============================================================================


class TestToolDefinition:
    """Tests for ToolDefinition Pydantic model."""

    # -- happy path --

    def test_defaults(self):
        # Arrange / Act
        tool = ToolDefinition(
            name="my-tool",
            container=ContainerConfig(image="img:v1", command="run")
        )

        # Assert
        assert tool.name == "my-tool"
        assert tool.is_baseline is False

    def test_baseline_flag(self):
        tool = ToolDefinition(
            name="noop",
            container=ContainerConfig(image="img:v1", command="run"),
            is_baseline=True
        )
        assert tool.is_baseline is True

    def test_container_nested_access(self):
        tool = ToolDefinition(
            name="t",
            container=ContainerConfig(image="my/img:latest", command="cmd", runtime="apptainer")
        )
        assert tool.container.image == "my/img:latest"
        assert tool.container.runtime == "apptainer"

    # -- failure modes --

    def test_missing_name_raises(self):
        with pytest.raises(Exception):
            ToolDefinition(container=ContainerConfig(image="img", command="cmd"))  # type: ignore[call-arg]

    def test_missing_container_raises(self):
        with pytest.raises(Exception):
            ToolDefinition(name="tool")  # type: ignore[call-arg]


# ============================================================================
# Tests: load_tools_from_yaml
# ============================================================================


class TestLoadToolsFromYaml:
    """Tests for load_tools_from_yaml function."""

    # -- happy path --

    def test_loads_all_tools(self, minimal_tools_yaml: Path):
        # Arrange / Act
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        # Assert
        assert "pre_noop" in tools
        assert "pre_tool" in tools
        assert len(tools) == 2

    def test_baseline_flag_propagates(self, minimal_tools_yaml: Path):
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        assert tools["pre_noop"].is_baseline is True
        assert tools["pre_tool"].is_baseline is False

    def test_returns_tool_definition_instances(self, minimal_tools_yaml: Path):
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        for tool in tools.values():
            assert isinstance(tool, ToolDefinition)

    def test_container_image_loaded(self, minimal_tools_yaml: Path):
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        assert tools["pre_noop"].container.image == "test/noop:v1"
        assert tools["pre_tool"].container.image == "test/pre_tool:v2"

    def test_container_command_loaded(self, minimal_tools_yaml: Path):
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        assert tools["pre_noop"].container.command == "python main.py"
        assert tools["pre_tool"].container.command == "python3 run.py"

    def test_tool_name_from_yaml(self, minimal_tools_yaml: Path):
        # 'name' in YAML overrides the key when present
        tools = load_tools_from_yaml(str(minimal_tools_yaml))

        assert tools["pre_noop"].name == "noop"
        assert tools["pre_tool"].name == "pre-tool"

    def test_runtime_loaded(self, multi_tools_yaml: Path):
        tools = load_tools_from_yaml(str(multi_tools_yaml))

        assert tools["deploy_tool"].container.runtime == "docker"
        assert tools["in_dp"].container.runtime is None

    # -- failure modes --

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError, match="Tools configuration file not found"):
            load_tools_from_yaml("/nonexistent/path/tools.yaml")

    def test_missing_tools_key_raises(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("something_else:\n  key: val\n")

        with pytest.raises(ValueError, match="Invalid tools.yaml format"):
            load_tools_from_yaml(str(bad_yaml))

    def test_empty_file_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")

        with pytest.raises(ValueError, match="Invalid tools.yaml format"):
            load_tools_from_yaml(str(empty))

    # -- edge cases --

    def test_default_name_falls_back_to_key(self, tmp_path: Path):
        # When 'name' field is absent, key is used
        yaml_content = """\
tools:
  my_key_tool:
    is_baseline: false
    container:
      image: img:v1
      command: run
"""
        f = tmp_path / "tools.yaml"
        f.write_text(yaml_content)
        tools = load_tools_from_yaml(str(f))
        assert tools["my_key_tool"].name == "my_key_tool"

    def test_default_is_baseline_false(self, tmp_path: Path):
        yaml_content = """\
tools:
  tool_no_flag:
    name: tool-no-flag
    container:
      image: img:v1
      command: run
"""
        f = tmp_path / "tools.yaml"
        f.write_text(yaml_content)
        tools = load_tools_from_yaml(str(f))
        assert tools["tool_no_flag"].is_baseline is False


# ============================================================================
# Tests: init_tool_registry
# ============================================================================


class TestInitToolRegistry:
    """Tests for init_tool_registry function."""

    # -- happy path --

    def test_populates_registry(self, minimal_tools_yaml: Path):
        init_tool_registry(str(minimal_tools_yaml))

        assert len(tools_module._TOOL_REGISTRY) == 2
        assert "pre_noop" in tools_module._TOOL_REGISTRY

    def test_replaces_previous_registry(self, minimal_tools_yaml: Path, tmp_path: Path):
        # Arrange: registry already has something
        tools_module._TOOL_REGISTRY["stale_tool"] = ToolDefinition(
            name="stale",
            container=ContainerConfig(image="img", command="cmd")
        )

        # Act: re-initialize with a fresh file
        init_tool_registry(str(minimal_tools_yaml))

        # Assert: old entry gone, new entries present
        assert "stale_tool" not in tools_module._TOOL_REGISTRY
        assert "pre_noop" in tools_module._TOOL_REGISTRY

    def test_multi_tool_yaml_all_loaded(self, multi_tools_yaml: Path):
        init_tool_registry(str(multi_tools_yaml))

        assert "pre_noop" in tools_module._TOOL_REGISTRY
        assert "in_dp" in tools_module._TOOL_REGISTRY
        assert "deploy_tool" in tools_module._TOOL_REGISTRY


# ============================================================================
# Tests: get_tool
# ============================================================================


class TestGetTool:
    """Tests for get_tool registry lookup."""

    # -- happy path --

    def test_returns_tool_for_known_name(self, minimal_tools_yaml: Path):
        init_tool_registry(str(minimal_tools_yaml))
        tool = get_tool("pre_noop")

        assert tool is not None
        assert tool.name == "noop"

    # -- failure modes --

    def test_returns_none_for_unknown_name(self, minimal_tools_yaml: Path):
        init_tool_registry(str(minimal_tools_yaml))

        result = get_tool("no_such_tool")
        assert result is None

    def test_returns_none_when_registry_empty(self):
        result = get_tool("anything")
        assert result is None

    # -- edge cases --

    def test_returns_correct_baseline_flag(self, minimal_tools_yaml: Path):
        init_tool_registry(str(minimal_tools_yaml))

        noop = get_tool("pre_noop")
        actual = get_tool("pre_tool")

        assert noop.is_baseline is True
        assert actual.is_baseline is False


# ============================================================================
# Tests: get_all_tools
# ============================================================================


class TestGetAllTools:
    """Tests for get_all_tools function."""

    # -- happy path --

    def test_returns_all_registered_tools(self, minimal_tools_yaml: Path):
        init_tool_registry(str(minimal_tools_yaml))
        tools = get_all_tools()

        assert len(tools) == 2
        assert "pre_noop" in tools
        assert "pre_tool" in tools

    def test_empty_when_registry_empty(self):
        tools = get_all_tools()
        assert tools == {}

    # -- edge cases --

    def test_returns_copy_not_reference(self, minimal_tools_yaml: Path):
        # Mutating the returned dict must not affect the registry
        init_tool_registry(str(minimal_tools_yaml))
        tools = get_all_tools()
        tools["injected"] = ToolDefinition(
            name="x",
            container=ContainerConfig(image="i", command="c")
        )

        assert "injected" not in tools_module._TOOL_REGISTRY
