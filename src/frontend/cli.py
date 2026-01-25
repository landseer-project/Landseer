"""CLI for Landseer Frontend."""

import subprocess
import sys
from pathlib import Path

# Frontend directory
FRONTEND_DIR = Path(__file__).parent


def install():
    """Install frontend dependencies."""
    print("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    print("Dependencies installed successfully!")


def dev():
    """Start the development server."""
    print("Starting development server...")
    print("Frontend will be available at http://localhost:3000")
    print("Make sure the backend API is running at http://localhost:8000")
    subprocess.run(["npm", "run", "dev"], cwd=FRONTEND_DIR)


def build():
    """Build the frontend for production."""
    print("Building frontend for production...")
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    print("Build complete! Output in dist/")


def preview():
    """Preview the production build."""
    print("Starting preview server...")
    subprocess.run(["npm", "run", "preview"], cwd=FRONTEND_DIR)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Landseer Frontend CLI")
        print("")
        print("Usage: python -m src.frontend.cli <command>")
        print("")
        print("Commands:")
        print("  install   Install dependencies")
        print("  dev       Start development server")
        print("  build     Build for production")
        print("  preview   Preview production build")
        return

    command = sys.argv[1]

    if command == "install":
        install()
    elif command == "dev":
        dev()
    elif command == "build":
        build()
    elif command == "preview":
        preview()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
