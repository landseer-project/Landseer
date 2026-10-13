"""CLI for Landseer Frontend."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

# Frontend directory
FRONTEND_DIR = Path(__file__).parent


def _require_npm() -> str:
    npm = shutil.which("npm")
    if not npm:
        click.echo(
            "Error: `npm` not found. Install Node.js 18+ (includes npm), then:\n"
            "  cd src/frontend && npm install && npm run dev\n"
            "Or from the repo root: uv run landseer-frontend install && "
            "uv run landseer-frontend dev",
            err=True,
        )
        raise SystemExit(1)
    return npm


def install() -> None:
    """Install frontend dependencies."""
    _require_npm()
    click.echo("Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    click.echo("Dependencies installed successfully!")


def dev(extra_args: Optional[list] = None) -> None:
    """Start the development server."""
    _require_npm()
    if extra_args is None:
        extra_args = []
    click.echo("Starting development server...")
    click.echo(
        "Frontend will be available at http://localhost:5173 (or custom --host/--port)"
    )
    click.echo("Make sure the backend API is running at http://localhost:8000")
    cmd = ["npm", "run", "dev"]
    if extra_args:
        cmd.extend(["--", *extra_args])
    subprocess.run(cmd, cwd=FRONTEND_DIR)


def build() -> None:
    """Build the frontend for production."""
    _require_npm()
    click.echo("Building frontend for production...")
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    click.echo("Build complete! Output in dist/")


def preview() -> None:
    """Preview the production build."""
    _require_npm()
    click.echo("Starting preview server...")
    subprocess.run(["npm", "run", "preview"], cwd=FRONTEND_DIR)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Landseer Frontend CLI."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("install")
def install_cmd() -> None:
    """Install frontend dependencies."""
    install()


@cli.command(
    "dev",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def dev_cmd(ctx: click.Context) -> None:
    """Start the development server."""
    extra_args = list(ctx.args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    dev(extra_args)


@cli.command("build")
def build_cmd() -> None:
    """Build the frontend for production."""
    build()


@cli.command("preview")
def preview_cmd() -> None:
    """Preview the production build."""
    preview()


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    try:
        result = cli.main(
            args=argv,
            prog_name="landseer-frontend",
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
