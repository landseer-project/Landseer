"""Command-line interface for Landseer worker."""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the worker CLI."""
    parser = argparse.ArgumentParser(
        description="Landseer Worker - ML Security Pipeline Task Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    
    parser.add_argument(
        "--worker-id",
        type=str,
        help="Unique worker identifier (auto-generated if not provided)",
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent tasks to process (default: 1)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the worker CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    print(f"Starting Landseer Worker...")
    print(f"Backend URL: {args.backend_url}")
    print(f"Worker ID: {args.worker_id or 'auto-generated'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Log Level: {args.log_level}")
    print(f"Debug: {args.debug}")
    
    if args.config:
        print(f"Config: {args.config}")
    
    # TODO: Implement worker startup logic
    print("Worker not yet implemented.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
