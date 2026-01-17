"""Command-line interface for Landseer frontend."""

import argparse
import sys
from typing import Optional


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the frontend CLI."""
    parser = argparse.ArgumentParser(
        description="Landseer Frontend - Machine Learning Security Pipeline UI",
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
        help="Host to bind the frontend server (default: 0.0.0.0)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to bind the frontend server (default: 3000)",
    )
    
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
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
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the frontend CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    print(f"Starting Landseer Frontend...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Backend URL: {args.backend_url}")
    print(f"Debug: {args.debug}")
    
    if args.config:
        print(f"Config: {args.config}")
    
    # TODO: Implement frontend server startup logic
    print("Frontend server not yet implemented.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
