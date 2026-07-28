"""CLI for Landseer Frontend."""

import shutil
import subprocess
import sys
from pathlib import Path

# Frontend directory
FRONTEND_DIR = Path(__file__).parent


def _require_npm() -> str:
    npm = shutil.which("npm")
    if not npm:
        print(
            "Error: `npm` not found. Install Node.js 18+ (includes npm), then:\n"
            "  cd src/frontend && npm install && npm run dev\n"
            "Or from the repo root: poetry run landseer-frontend install && poetry run landseer-frontend dev",
            file=sys.stderr,
        )
        sys.exit(1)
    return npm


def install():
    """Install frontend dependencies."""
    _require_npm()
    print("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    print("Dependencies installed successfully!")


def dev(extra_args=None):
    """Start the development server."""
    _require_npm()
    if extra_args is None:
        extra_args = []
    print("Starting development server...")
    print("Frontend will be available at http://localhost:5173 (or custom --host/--port)")
    print("Make sure the backend API is running at http://localhost:8000")
    cmd = ["npm", "run", "dev"]
    if extra_args:
        cmd.extend(["--", *extra_args])
    subprocess.run(cmd, cwd=FRONTEND_DIR)


def build():
    """Build the frontend for production."""
    _require_npm()
    print("Building frontend for production...")
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    print("Build complete! Output in dist/")


def preview():
    """Preview the production build."""
    _require_npm()
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

    command_args = sys.argv[2:]
    if command_args[:1] == ["--"]:
        command_args = command_args[1:]

    if command == "install":
        install()
    elif command == "dev":
        dev(command_args)
    elif command == "build":
        build()
    elif command == "preview":
        preview()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
