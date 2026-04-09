"""
Unit tests for backend CLI argument parsing.

Tests cover:
- create_parser returns a valid ArgumentParser
- --config defaults to None (headless mode)
- --config accepts a path when specified
- --default-pipeline argument no longer exists
- --tools-config has correct default
- --host / --port / --debug defaults
- main() headless path (pipeline_config_path=None forwarded to initialize_backend)
- main() with --config path (pipeline_config_path forwarded)
"""

import argparse
import pytest
from unittest.mock import patch, MagicMock

from src.backend.cli import create_parser, main


class TestCreateParser:
    """Tests for create_parser()."""

    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_config_defaults_to_none(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.config is None

    def test_config_accepts_path(self):
        parser = create_parser()
        args = parser.parse_args(["--config", "configs/pipeline/mini.yaml"])
        assert args.config == "configs/pipeline/mini.yaml"

    def test_default_pipeline_arg_removed(self):
        parser = create_parser()
        assert not hasattr(parser.parse_args([]), "default_pipeline")

    def test_tools_config_default(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.tools_config == "configs/tools.yaml"

    def test_tools_config_override(self):
        parser = create_parser()
        args = parser.parse_args(["--tools-config", "/custom/tools.yaml"])
        assert args.tools_config == "/custom/tools.yaml"

    def test_host_default(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.host == "0.0.0.0"

    def test_port_default(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.port == 8000

    def test_debug_default_false(self):
        parser = create_parser()
        args = parser.parse_args([])
        assert args.debug is False

    def test_debug_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--debug"])
        assert args.debug is True


class TestMainHeadless:
    """Tests for main() in headless mode (no --config)."""

    @patch("src.backend.cli.initialize_backend")
    @patch("src.backend.cli.set_backend_context")
    def test_headless_passes_none_config(self, mock_set_ctx, mock_init):
        mock_ctx = MagicMock()
        mock_ctx.pipeline = None
        mock_init.return_value = mock_ctx

        with patch("src.backend.api.run_server", side_effect=KeyboardInterrupt):
            main([])

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs.get("pipeline_config_path") is None or \
               call_kwargs[1].get("pipeline_config_path") is None

    @patch("src.backend.cli.initialize_backend")
    @patch("src.backend.cli.set_backend_context")
    def test_with_config_passes_path(self, mock_set_ctx, mock_init):
        mock_ctx = MagicMock()
        mock_ctx.pipeline = MagicMock()
        mock_ctx.pipeline.name = "test"
        mock_ctx.pipeline.workflows = []
        mock_init.return_value = mock_ctx

        with patch("src.backend.api.run_server", side_effect=KeyboardInterrupt):
            main(["--config", "configs/pipeline/mini.yaml"])

        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs.get("pipeline_config_path") == "configs/pipeline/mini.yaml" or \
               call_kwargs[1].get("pipeline_config_path") == "configs/pipeline/mini.yaml"

    @patch("src.backend.cli.initialize_backend")
    @patch("src.backend.cli.set_backend_context")
    def test_headless_context_set(self, mock_set_ctx, mock_init):
        mock_ctx = MagicMock()
        mock_ctx.pipeline = None
        mock_init.return_value = mock_ctx

        with patch("src.backend.api.run_server", side_effect=KeyboardInterrupt):
            result = main([])

        mock_set_ctx.assert_called_once_with(mock_ctx)
        assert result == 0

    @patch("src.backend.cli.initialize_backend", side_effect=FileNotFoundError("not found"))
    def test_file_not_found_returns_1(self, mock_init):
        result = main(["--config", "/nonexistent.yaml"])
        assert result == 1

    @patch("src.backend.cli.initialize_backend", side_effect=ValueError("bad config"))
    def test_value_error_returns_1(self, mock_init):
        result = main(["--config", "/bad.yaml"])
        assert result == 1
