"""
Adapter layers for tools that do not use Landseer's standard formats.

A tool declared with adapters in tools.yaml (see docs/how-to/adapter-layers.md)
runs as one task of up to three containers:

    input adapter  ->  tool  ->  output adapter

Each container is started by the same runner (Docker or Apptainer) with the
same mounts and environment as a plain tool, so both runtimes behave alike.
The task workspace is shared between them:

- input adapter: /input = task input, /output = <task>/adapters/input.
  Its files are then overlaid onto the task input, so the tool sees the
  standard inputs plus the converted ones (e.g. model_tf/model.keras).
- tool: /input = task input, /output = task output (unchanged contract).
- output adapter: /input = task input with the tool's outputs overlaid,
  /output = <task>/adapters/output. Its files are then overlaid onto the task
  output, replacing tool files with the same name (e.g. model.pt).

The adapters are carried in the task config (``config["adapters"]``), so they
are part of the task's hash, lineage and cache key.
"""

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..common import get_logger
from ..common.fsutil import mirror_tree_link_or_copy

logger = get_logger(__name__)

ADAPTER_POSITIONS = ("input", "output")


def task_adapters(config: Any) -> Dict[str, Dict[str, Any]]:
    """
    Return the adapters declared in a task config, keyed by position.

    Raises:
        ValueError: If ``config["adapters"]`` is present but malformed.
    """
    raw = config.get("adapters") if isinstance(config, dict) else None
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Task config 'adapters' must be a mapping")
    adapters: Dict[str, Dict[str, Any]] = {}
    for position in ADAPTER_POSITIONS:
        spec = raw.get(position)
        if not spec:
            continue
        if not isinstance(spec, dict) or not spec.get("image"):
            raise ValueError(f"{position} adapter in task config must have an image")
        adapters[position] = spec
    return adapters


def run_adapter_chain(
    container_runner: Any,
    *,
    adapters: Dict[str, Dict[str, Any]],
    tool_image: str,
    tool_command: str,
    task_dir: Path,
    input_dir: Path,
    output_dir: Path,
    env: Optional[Dict[str, str]] = None,
    extra_mounts: Optional[Dict[str, str]] = None,
    model_script_path: Optional[Path] = None,
    pull_image: Optional[Callable[[str], bool]] = None,
) -> Tuple[int, str, str, Optional[str]]:
    """
    Run input adapter -> tool -> output adapter in one task workspace.

    Args:
        container_runner: DockerRunner or ApptainerRunner
        adapters: Output of :func:`task_adapters`
        tool_image: Tool container image
        tool_command: Tool command (already normalized by the caller)
        task_dir: Task workspace root; adapter outputs go under ``task_dir/adapters``
        input_dir: Task input directory (gets the input adapter's files)
        output_dir: Task output directory (gets the output adapter's files)
        env: Container environment shared by all three containers
        extra_mounts: Extra mounts shared by all three containers
        model_script_path: Model config script mounted into all three containers
        pull_image: Called for each adapter image before it runs (Docker only)

    Returns:
        Tuple of (exit_code, logs, commands, error_message). error_message
        names the failing step and is None on success.
    """
    logs: List[str] = []
    commands: List[str] = []

    def _run(label: str, image: str, command: str, in_dir: Path, out_dir: Path, run_env: Dict[str, str]) -> int:
        logger.info(f"Running {label}: {image}")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = container_runner.run(
            image=image,
            command=command,
            input_dir=in_dir,
            output_dir=out_dir,
            # Runners may add INPUT_DIR/OUTPUT_DIR to the dict they get; keep ours clean.
            env=dict(run_env),
            extra_mounts=extra_mounts,
            model_script_path=model_script_path,
        )
        if len(result) == 3:
            exit_code, out, cmd = result
        else:
            exit_code, out = result
            cmd = None
        logs.append(f"===== {label} ({image}) exit={exit_code} =====\n{out}")
        if cmd:
            commands.append(cmd)
        return exit_code

    def _done(exit_code: int, error_message: Optional[str]) -> Tuple[int, str, str, Optional[str]]:
        if error_message:
            logs.append(error_message)
        return exit_code, "\n".join(logs), "\n".join(commands), error_message

    def _adapter_env(position: str, spec: Dict[str, Any]) -> Dict[str, str]:
        adapter_env = dict(env or {})
        adapter_env.update({str(k): str(v) for k, v in (spec.get("env") or {}).items()})
        adapter_env["LANDSEER_ADAPTER"] = position
        adapter_env["LANDSEER_TOOL_IMAGE"] = tool_image
        return adapter_env

    adapter_root = Path(task_dir) / "adapters"
    if adapter_root.exists():
        shutil.rmtree(adapter_root, ignore_errors=True)

    spec = adapters.get("input")
    if spec:
        name = spec.get("name") or spec["image"]
        if pull_image is not None and not pull_image(spec["image"]):
            return _done(-1, f"Failed to pull input adapter image: {spec['image']}")
        stage_out = adapter_root / "input"
        exit_code = _run(
            f"input adapter '{name}'", spec["image"], spec.get("command") or "",
            input_dir, stage_out, _adapter_env("input", spec),
        )
        if exit_code != 0:
            return _done(exit_code, f"Input adapter '{name}' exited with code {exit_code}")
        mirror_tree_link_or_copy(stage_out, input_dir)
        shutil.rmtree(stage_out, ignore_errors=True)

    exit_code = _run("tool", tool_image, tool_command, input_dir, output_dir, dict(env or {}))
    if exit_code != 0:
        return _done(exit_code, f"Container exited with code {exit_code}")

    spec = adapters.get("output")
    if spec:
        name = spec.get("name") or spec["image"]
        if pull_image is not None and not pull_image(spec["image"]):
            return _done(-1, f"Failed to pull output adapter image: {spec['image']}")
        # The output adapter reads the tool's outputs next to the task inputs
        # (config_model.py, data, the input adapter's files).
        mirror_tree_link_or_copy(output_dir, input_dir)
        stage_out = adapter_root / "output"
        exit_code = _run(
            f"output adapter '{name}'", spec["image"], spec.get("command") or "",
            input_dir, stage_out, _adapter_env("output", spec),
        )
        if exit_code != 0:
            return _done(exit_code, f"Output adapter '{name}' exited with code {exit_code}")
        mirror_tree_link_or_copy(stage_out, output_dir)
        shutil.rmtree(stage_out, ignore_errors=True)

    # Staging dirs are copied out above; a failed step leaves its own for debugging.
    if adapter_root.exists() and not any(adapter_root.iterdir()):
        adapter_root.rmdir()
    return _done(0, None)
