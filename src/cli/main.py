"""Landseer CLI
Talks to a running backend over HTTP. 
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import click

from .client import LandseerApiClient, LandseerApiError

def _echo_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


def _echo_kv(data: Dict[str, Any], keys: Optional[List[str]] = None) -> None:
    items = keys or list(data.keys())
    width = max((len(k) for k in items), default=0)
    for key in items:
        if key not in data:
            continue
        click.echo(f"{key.ljust(width)}  {data[key]}")


def _echo_table(rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        click.echo("(none)")
        return
    widths = {
        col: max(len(col), *(len(str(r.get(col, ""))) for r in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    click.echo(header)
    click.echo("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        click.echo(
            "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        )


def _client_from_ctx(ctx: click.Context) -> LandseerApiClient:
    return ctx.obj["client"]


def _want_json(ctx: click.Context) -> bool:
    return bool(ctx.obj.get("as_json"))


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Landseer control-plane CLI — submit runs and inspect a running backend.",
)
@click.option(
    "--backend-url",
    default=lambda: os.environ.get("LANDSEER_BACKEND_URL", "http://localhost:8000"),
    show_default="http://localhost:8000",
    help="Backend API URL (env: LANDSEER_BACKEND_URL)",
)
@click.option(
    "--pipeline-key",
    default=None,
    envvar="LANDSEER_PIPELINE_KEY",
    help="X-Pipeline-Key for mutating endpoints (env: LANDSEER_PIPELINE_KEY)",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON",
)
@click.pass_context
def cli(
    ctx: click.Context,
    backend_url: str,
    pipeline_key: Optional[str],
    as_json: bool,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["as_json"] = as_json
    ctx.obj["client"] = LandseerApiClient(
        backend_url=backend_url,
        pipeline_key=pipeline_key,
    )

    @ctx.call_on_close
    def _close() -> None:
        ctx.obj["client"].close()


@cli.command("health")
@click.pass_context
def health_cmd(ctx: click.Context) -> None:
    """Check backend health."""
    client = _client_from_ctx(ctx)
    try:
        data = client.health()
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e
    if _want_json(ctx):
        _echo_json(data)
        return
    _echo_kv(data, ["status", "scheduler_active", "timestamp"])


@cli.command("status")
@click.option("--run-id", default=None, help="Filter progress to one run")
@click.pass_context
def status_cmd(ctx: click.Context, run_id: Optional[str]) -> None:
    """Compact stack status: health + progress + workers."""
    client = _client_from_ctx(ctx)
    try:
        health = client.health()
        progress = client.progress(run_id=run_id)
        workers = client.list_workers()
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e

    payload = {
        "health": health,
        "progress": progress,
        "workers": {
            "total": workers.get("total"),
            "active": workers.get("active"),
            "workers": workers.get("workers", []),
        },
    }
    if _want_json(ctx):
        _echo_json(payload)
        return

    click.echo("Health")
    _echo_kv(health, ["status", "scheduler_active", "timestamp"])
    click.echo("")
    click.echo("Progress" + (f" (run {run_id})" if run_id else ""))
    _echo_kv(
        progress,
        [
            "total",
            "pending",
            "running",
            "completed",
            "failed",
            "progress_percent",
            "is_complete",
        ],
    )
    click.echo("")
    click.echo(
        f"Workers  total={workers.get('total', 0)}  active={workers.get('active', 0)}"
    )


@cli.command("progress")
@click.option("--run-id", default=None, help="Filter progress to one run")
@click.pass_context
def progress_cmd(ctx: click.Context, run_id: Optional[str]) -> None:
    """Show scheduler progress."""
    client = _client_from_ctx(ctx)
    try:
        data = client.progress(run_id=run_id)
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e
    if _want_json(ctx):
        _echo_json(data)
        return
    _echo_kv(
        data,
        [
            "total",
            "pending",
            "running",
            "completed",
            "failed",
            "progress_percent",
            "is_complete",
        ],
    )


@cli.group("run", help="Submit and manage pipeline runs.")
def run_group() -> None:
    pass


@run_group.command("start")
@click.argument(
    "pipeline_yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
)
@click.pass_context
def run_start(ctx: click.Context, pipeline_yaml: str) -> None:
    """Submit a pipeline YAML path; backend builds tasks and schedules them."""
    client = _client_from_ctx(ctx)
    try:
        config_id = client.config_id_from_yaml(pipeline_yaml)
        data = client.start_run(config_id)
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e

    if _want_json(ctx):
        _echo_json(data)
        return

    click.echo(f"Submitted run {data.get('id')} for {config_id}")
    _echo_kv(
        data,
        [
            "id",
            "experiment_id",
            "run_number",
            "status",
            "use_cache",
            "created_at",
            "started_at",
            "error_message",
        ],
    )


@run_group.command("stop")
@click.argument("run_id")
@click.pass_context
def run_stop(ctx: click.Context, run_id: str) -> None:
    """Stop a pipeline run by run id."""
    client = _client_from_ctx(ctx)
    try:
        data = client.stop_run(run_id)
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e
    if _want_json(ctx):
        _echo_json(data)
        return
    click.echo(f"Stopped run {data.get('id')}")
    _echo_kv(data, ["id", "experiment_id", "status", "completed_at", "error_message"])


@run_group.command("list")
@click.pass_context
def run_list(ctx: click.Context) -> None:
    """List pipeline runs and their statuses."""
    client = _client_from_ctx(ctx)
    try:
        data = client.list_runs()
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e

    runs: List[Dict[str, Any]] = list(data.get("runs") or [])
    if _want_json(ctx):
        _echo_json({"runs": runs, "total": len(runs)})
        return

    _echo_table(
        runs,
        ["id", "experiment_id", "run_number", "status", "use_cache", "created_at"],
    )
    click.echo(f"\ntotal={len(runs)}")


@run_group.command("show")
@click.argument("run_id")
@click.pass_context
def run_show(ctx: click.Context, run_id: str) -> None:
    """Show one run's status and metadata."""
    client = _client_from_ctx(ctx)
    try:
        data = client.get_run(run_id)
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e
    if _want_json(ctx):
        _echo_json(data)
        return
    _echo_kv(
        data,
        [
            "id",
            "experiment_id",
            "run_number",
            "status",
            "use_cache",
            "created_at",
            "started_at",
            "completed_at",
            "error_message",
        ],
    )


@cli.group("task", help="Inspect pipeline tasks.")
def task_group() -> None:
    pass


@task_group.command("list")
@click.argument("run_id")
@click.option(
    "--status",
    "status_filter",
    default=None,
    type=click.Choice(["pending", "running", "completed", "failed"], case_sensitive=False),
    help="Optional status filter",
)
@click.pass_context
def task_list(ctx: click.Context, run_id: str, status_filter: Optional[str]) -> None:
    """List tasks for a run id."""
    client = _client_from_ctx(ctx)
    try:
        data = client.list_tasks(run_id, status=status_filter)
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e

    tasks: List[Dict[str, Any]] = list(data.get("tasks") or [])
    if _want_json(ctx):
        _echo_json(data)
        return

    rows = []
    for t in tasks:
        tool = t.get("tool") or {}
        rows.append(
            {
                "id": t.get("id"),
                "tool": tool.get("name") or "-",
                "status": t.get("status"),
                "type": t.get("task_type") or "-",
                "priority": t.get("priority", ""),
            }
        )
    _echo_table(rows, ["id", "tool", "status", "type", "priority"])
    click.echo(f"\ntotal={data.get('total', len(tasks))}")


@cli.group("worker", help="Inspect registered workers.")
def worker_group() -> None:
    pass


@worker_group.command("list")
@click.pass_context
def worker_list(ctx: click.Context) -> None:
    """List workers registered with the backend."""
    client = _client_from_ctx(ctx)
    try:
        data = client.list_workers()
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e

    workers: List[Dict[str, Any]] = list(data.get("workers") or [])
    if _want_json(ctx):
        _echo_json(data)
        return

    rows = [
        {
            "worker_id": w.get("worker_id"),
            "hostname": w.get("hostname"),
            "status": w.get("status"),
            "current_task_id": w.get("current_task_id") or "-",
            "completed": w.get("tasks_completed", 0),
            "failed": w.get("tasks_failed", 0),
        }
        for w in workers
    ]
    _echo_table(
        rows,
        ["worker_id", "hostname", "status", "current_task_id", "completed", "failed"],
    )
    click.echo(f"\ntotal={data.get('total', len(workers))}  active={data.get('active', 0)}")


@cli.group("config", help="Inspect discovered pipeline configs.")
def config_group() -> None:
    pass


@config_group.command("list")
@click.pass_context
def config_list(ctx: click.Context) -> None:
    """List pipeline configs known to the backend."""
    client = _client_from_ctx(ctx)
    try:
        data = client.list_configs()
    except LandseerApiError as e:
        raise click.ClickException(str(e)) from e
    configs: List[Dict[str, Any]] = list(data.get("configs") or [])
    if _want_json(ctx):
        _echo_json(data)
        return
    _echo_table(configs, ["id", "name", "config_path"])
    click.echo(f"\ntotal={data.get('total', len(configs))}")


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for `landseer`."""
    try:
        result = cli.main(args=argv, prog_name="landseer", standalone_mode=False)
        return int(result) if result is not None else 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except click.ClickException as e:
        e.show()
        return e.exit_code
    except LandseerApiError as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
