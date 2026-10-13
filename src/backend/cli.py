"""Command-line interface for Landseer backend."""

import sys
from types import SimpleNamespace
from typing import Optional

import click

from ..common import get_logger, init_sentry
from .initialization import initialize_backend, set_backend_context

logger = get_logger(__name__)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Landseer Backend - Machine Learning Security Pipeline Backend",
)
@click.version_option(version="0.1.0", prog_name="landseer-backend")
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind the backend server",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind the backend server",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode",
)
@click.option(
    "--config",
    "--pipeline-config",
    "config",
    default=None,
    type=click.Path(),
    help=(
        "Path to pipeline configuration file. If omitted, the backend starts "
        "without an active pipeline and waits for runs to be triggered via the API."
    ),
)
@click.option(
    "--tools-config",
    default="configs/tools.yaml",
    show_default=True,
    type=click.Path(),
    help="Path to tools configuration file",
)
@click.option(
    "--datasets-config",
    default="configs/datasets.yaml",
    show_default=True,
    type=click.Path(),
    help="Path to datasets configuration file",
)
@click.option(
    "--attack-config",
    default="configs/attacks.yaml",
    show_default=True,
    type=click.Path(),
    help="Path to attacks configuration file",
)
@click.option(
    "--data-dir",
    default="/data/landseer/landseer_old_data/data/",
    show_default=True,
    type=click.Path(),
    help="Path to the data directory",
)
def cli(
    host: str,
    port: int,
    debug: bool,
    config: Optional[str],
    tools_config: str,
    datasets_config: str,
    attack_config: str,
    data_dir: str,
) -> int:
    """Run the Landseer backend server."""
    # datasets_config / attack_config retained for CLI compatibility
    _ = (datasets_config, attack_config)

    init_sentry("backend")

    if debug:
        from ..common import set_global_log_level
        import logging

        set_global_log_level(logging.DEBUG)
        logger.debug("Debug mode enabled")

    logger.info("=" * 60)
    logger.info("Starting Landseer Backend...")
    logger.info("=" * 60)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")

    try:
        context = initialize_backend(
            tools_config_path=tools_config,
            pipeline_config_path=config,
            data_dir=data_dir,
        )
        set_backend_context(context)

        if context.pipeline is not None:
            logger.info(f"\nLoaded pipeline: {context.pipeline.name}")
            logger.info(f"Number of workflows: {len(context.pipeline.workflows)}")
            logger.info(
                f"Workflows: {', '.join([w.name for w in context.pipeline.workflows[:5]])}..."
            )
        else:
            logger.info("\nNo pipeline config specified, running in headless mode.")
            logger.info("Trigger runs via POST /api/pipeline-configs/<config_id>/runs")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}", exc_info=debug)
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("Backend initialized successfully!")
    logger.info("Starting FastAPI server...")
    logger.info("=" * 60)

    try:
        from .api import run_server

        run_server(
            host=host,
            port=port,
            reload=debug,
            log_level="debug" if debug else "info",
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=debug)
        return 1

    return 0


def parse_args(argv: Optional[list[str]] = None) -> SimpleNamespace:
    """Parse CLI arguments without running the server (for tests)."""
    with cli.make_context("landseer-backend", list(argv or [])) as ctx:
        return SimpleNamespace(**ctx.params)


def create_parser():
    """Test helper exposing parse_args(), matching the old argparse API."""

    class _Parser:
        def parse_args(self, argv: Optional[list[str]] = None) -> SimpleNamespace:
            return parse_args(argv)

    return _Parser()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the backend CLI."""
    try:
        result = cli.main(
            args=argv,
            prog_name="landseer-backend",
            standalone_mode=False,
        )
        return int(result) if result is not None else 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


if __name__ == "__main__":
    sys.exit(main())
