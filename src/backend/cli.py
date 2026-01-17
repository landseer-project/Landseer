"""Command-line interface for Landseer backend."""

import argparse
import sys
from typing import Optional

from ..common import get_logger
from .initialization import initialize_backend, set_backend_context

# Get logger for this module
logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the backend CLI."""
    parser = argparse.ArgumentParser(
        description="Landseer Backend - Machine Learning Security Pipeline Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the backend server (default: 0.0.0.0)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the backend server (default: 8000)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to pipeline configuration file (default: configs/pipeline/trades.yaml)",
    )
    
    parser.add_argument(
        "--tools-config",
        type=str,
        default="configs/tools.yaml",
        help="Path to tools configuration file (default: configs/tools.yaml)",
    )
    
    parser.add_argument(
        "--default-pipeline",
        type=str,
        default="trades",
        help="Default pipeline to load if --config not specified (default: trades)",
    )
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the backend CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Set logging level based on debug flag
    if args.debug:
        from ..common import set_global_log_level
        import logging
        set_global_log_level(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info("="*60)
    logger.info("Starting Landseer Backend...")
    logger.info("="*60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Debug: {args.debug}")
    
    # Initialize backend with configuration
    try:
        context = initialize_backend(
            tools_config_path=args.tools_config,
            pipeline_config_path=args.config,
            default_pipeline=args.default_pipeline
        )
        set_backend_context(context)
        
        logger.info(f"\nLoaded pipeline: {context.pipeline.name}")
        logger.info(f"Number of workflows: {len(context.pipeline.workflows)}")
        logger.info(f"Workflows: {', '.join([w.name for w in context.pipeline.workflows[:5]])}...")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}", exc_info=args.debug)
        return 1
    
    # TODO: Implement backend server startup logic (FastAPI, Flask, etc.)
    logger.info("\n" + "="*60)
    logger.info("Backend initialized successfully!")
    logger.info("Backend server implementation pending.")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
