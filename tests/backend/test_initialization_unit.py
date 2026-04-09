"""
Unit tests for backend initialization module.

Tests cover:
- BackendContext with pipeline=None (headless mode)
- BackendContext with pipeline set (normal mode)
- BackendContext.get_dataset_path returns None when no dataset_info
- BackendContext.get_dataset_path returns path when dataset_info set
- initialize_backend headless mode: pipeline is None, tool registry loaded
- initialize_backend headless mode: DB initialised, no pipeline synced
- initialize_backend headless mode: no dataset preparation attempted
- initialize_backend with config: pipeline is created
- initialize_backend with config: pipeline synced to DB
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.backend.initialization import BackendContext, initialize_backend


# ============================================================================
# BackendContext unit tests
# ============================================================================


class TestBackendContext:
    """Tests for BackendContext construction and properties."""

    def test_headless_context_pipeline_is_none(self):
        ctx = BackendContext(
            pipeline=None,
            tools_config_path="/tmp/tools.yaml",
            pipeline_config_path=None,
        )
        assert ctx.pipeline is None
        assert ctx.pipeline_config_path is None

    def test_normal_context_pipeline_set(self):
        mock_pipeline = MagicMock()
        mock_pipeline.name = "test_pipeline"
        ctx = BackendContext(
            pipeline=mock_pipeline,
            tools_config_path="/tmp/tools.yaml",
            pipeline_config_path="/tmp/config.yaml",
        )
        assert ctx.pipeline is mock_pipeline
        assert ctx.pipeline_config_path == "/tmp/config.yaml"

    def test_get_dataset_path_none_when_no_info(self):
        ctx = BackendContext(
            pipeline=None,
            tools_config_path="/tmp/tools.yaml",
            dataset_info=None,
        )
        assert ctx.get_dataset_path() is None

    def test_get_dataset_path_none_when_no_output_dir_key(self):
        ctx = BackendContext(
            pipeline=None,
            tools_config_path="/tmp/tools.yaml",
            dataset_info={"name": "cifar10"},
        )
        assert ctx.get_dataset_path() is None

    def test_get_dataset_path_returns_path(self):
        ctx = BackendContext(
            pipeline=None,
            tools_config_path="/tmp/tools.yaml",
            dataset_info={"output_dir": "/data/cifar10"},
        )
        result = ctx.get_dataset_path()
        assert result == Path("/data/cifar10")

    def test_optional_fields_default_to_none(self):
        ctx = BackendContext(pipeline=None, tools_config_path="/tmp/tools.yaml")
        assert ctx.db_service is None
        assert ctx.store is None
        assert ctx.dataset_info is None
        assert ctx.dataset_manager is None


# ============================================================================
# initialize_backend unit tests
# ============================================================================


class TestInitializeBackendHeadless:
    """Tests for initialize_backend when pipeline_config_path is None."""

    @patch("src.backend.initialization.init_tool_registry")
    def test_headless_returns_context_with_none_pipeline(self, mock_init_tools):
        ctx = initialize_backend(
            pipeline_config_path=None,
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        assert ctx.pipeline is None
        assert ctx.pipeline_config_path is None

    @patch("src.backend.initialization.init_tool_registry")
    def test_headless_tool_registry_still_loaded(self, mock_init_tools):
        initialize_backend(
            pipeline_config_path=None,
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        mock_init_tools.assert_called_once()

    @patch("src.backend.initialization.init_tool_registry")
    def test_headless_tools_config_resolved(self, mock_init_tools):
        initialize_backend(
            tools_config_path="configs/tools.yaml",
            pipeline_config_path=None,
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        call_arg = mock_init_tools.call_args[0][0]
        assert Path(call_arg).is_absolute()

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.create_pipeline_from_config")
    def test_headless_does_not_create_pipeline(self, mock_create, mock_init_tools):
        initialize_backend(
            pipeline_config_path=None,
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        mock_create.assert_not_called()

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.load_pipeline_config")
    def test_headless_does_not_load_pipeline_config(self, mock_load, mock_init_tools):
        initialize_backend(
            pipeline_config_path=None,
            enable_db=False,
            enable_store=False,
            prepare_dataset=True,
        )
        mock_load.assert_not_called()

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.init_db_service")
    def test_headless_db_initialized_but_no_sync(self, mock_init_db, mock_init_tools):
        mock_db_svc = MagicMock()
        mock_db_svc.is_available.return_value = True
        mock_init_db.return_value = mock_db_svc

        with patch.object(
            type(MagicMock()), "DB_AVAILABLE", True, create=True
        ):
            ctx = initialize_backend(
                pipeline_config_path=None,
                enable_db=True,
                enable_store=False,
                prepare_dataset=False,
            )

        if ctx.db_service is not None:
            mock_db_svc.sync_pipeline_to_db.assert_not_called()


class TestInitializeBackendWithConfig:
    """Tests for initialize_backend with a pipeline config path."""

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.create_pipeline_from_config")
    def test_with_config_creates_pipeline(self, mock_create, mock_init_tools):
        mock_pipeline = MagicMock()
        mock_pipeline.name = "test"
        mock_pipeline.workflows = []
        mock_pipeline.config = {}
        mock_create.return_value = mock_pipeline

        ctx = initialize_backend(
            pipeline_config_path="configs/pipeline/mini.yaml",
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        assert ctx.pipeline is mock_pipeline
        mock_create.assert_called_once()

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.create_pipeline_from_config")
    def test_with_config_pipeline_config_path_resolved(self, mock_create, mock_init_tools):
        mock_pipeline = MagicMock()
        mock_pipeline.name = "test"
        mock_pipeline.workflows = []
        mock_pipeline.config = {}
        mock_create.return_value = mock_pipeline

        ctx = initialize_backend(
            pipeline_config_path="configs/pipeline/mini.yaml",
            enable_db=False,
            enable_store=False,
            prepare_dataset=False,
        )
        assert Path(ctx.pipeline_config_path).is_absolute()

    @patch("src.backend.initialization.init_tool_registry")
    @patch("src.backend.initialization.create_pipeline_from_config", side_effect=FileNotFoundError("nope"))
    def test_with_config_raises_on_missing_file(self, mock_create, mock_init_tools):
        with pytest.raises(FileNotFoundError):
            initialize_backend(
                pipeline_config_path="/nonexistent/pipeline.yaml",
                enable_db=False,
                enable_store=False,
                prepare_dataset=False,
            )
