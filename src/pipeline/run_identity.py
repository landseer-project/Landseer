"""
Run-scoped identity for tasks and workflows.

Tasks are built with provisional counter IDs (``task_N``) that restart with
every backend process, so they collide across runs. When a pipeline is bound
to a run, every task and workflow gets an ID that is unique across runs:

    task.id      = "<short_run_id>-<lineage_hash[:12]>"
    workflow.id  = "<short_run_id>-<workflow.name>"

``lineage_hash`` identifies the computation: the tool, the task's own config
and, transitively, the lineage of every dependency. Within one run the same
lineage means the same task (the config loader already shares such tasks
between workflows); across runs the run prefix keeps IDs apart.

The worker still computes the content-addressed cache key at execution time,
because only it knows the dataset content and model script actually used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional

from .pipeline import Pipeline
from .tasks import EvaluationTask, Task

logger = logging.getLogger(__name__)

# Config keys with this prefix are set per run (e.g. "_run_use_cache") and are
# not part of what the task computes.
RUN_SCOPED_CONFIG_PREFIX = "_run_"

SHORT_RUN_ID_LEN = 8
LINEAGE_ID_LEN = 12

# "e807ab0e-3f9a2c41d0e7", optionally "-2" on the (astronomically unlikely)
# event of a 48-bit prefix collision within a run.
RUN_SCOPED_TASK_ID_RE = re.compile(
    rf"^[0-9a-f]{{{SHORT_RUN_ID_LEN}}}-[0-9a-f]{{{LINEAGE_ID_LEN}}}(-\d+)?$"
)

_RUN_ID_SUFFIX_RE = re.compile(rf"_([0-9a-f]{{{SHORT_RUN_ID_LEN}}})$")


def short_run_id(run_id: str) -> str:
    """
    Short, stable prefix for a run.

    Run IDs look like ``run_20261013_232027_e807ab0e``; their random hex suffix
    is used directly. Any other ID shape is hashed.
    """
    m = _RUN_ID_SUFFIX_RE.search(run_id)
    if m:
        return m.group(1)
    return hashlib.sha256(run_id.encode()).hexdigest()[:SHORT_RUN_ID_LEN]


def lineage_hash(task: Task, memo: Optional[Dict[int, str]] = None) -> str:
    """sha256 over the task's tool, own config and its dependencies' lineage."""
    memo = {} if memo is None else memo
    key = id(task)
    if key in memo:
        return memo[key]
    data = {
        "tool_name": task.tool.name,
        "tool_image": task.tool.container.image,
        "tool_command": task.tool.container.command,
        "config": {
            k: v for k, v in (task.config or {}).items()
            if not k.startswith(RUN_SCOPED_CONFIG_PREFIX)
        },
        "dependencies": sorted(lineage_hash(dep, memo) for dep in task.dependencies),
    }
    if isinstance(task, EvaluationTask):
        data["required_artifacts"] = sorted(task.required_artifacts)
    digest = hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()
    memo[key] = digest
    return digest


def _unique_tasks(pipeline: Pipeline) -> List[Task]:
    """All tasks of the pipeline once each, dependencies included."""
    seen: Dict[int, Task] = {}
    stack = [t for wf in pipeline.workflows for t in wf.tasks]
    while stack:
        task = stack.pop()
        if id(task) in seen:
            continue
        seen[id(task)] = task
        stack.extend(task.dependencies)
    return list(seen.values())


def bind_pipeline_to_run(
    pipeline: Pipeline,
    run_id: str,
    use_cache: Optional[bool] = None,
) -> None:
    """
    Give the pipeline, its workflows and its tasks run-scoped IDs.

    Args:
        pipeline: Freshly built pipeline (not yet scheduled or synced to the DB).
        run_id: Experiment run ID the pipeline executes as.
        use_cache: If given, stored as ``config["_run_use_cache"]`` on every task
            so workers apply the run's cache policy.
    """
    prefix = short_run_id(run_id)
    pipeline.id = run_id

    workflow_ids: Dict[str, str] = {}
    used_workflow_ids: set = set()
    for workflow in pipeline.workflows:
        new_id = f"{prefix}-{workflow.name}"
        n = 2
        while new_id in used_workflow_ids:
            new_id = f"{prefix}-{workflow.name}-{n}"
            n += 1
        used_workflow_ids.add(new_id)
        workflow_ids[workflow.id] = new_id
        workflow.id = new_id
        workflow.pipeline_id = run_id
        workflow.run_id = run_id

    memo: Dict[int, str] = {}
    used_task_ids: Dict[str, Task] = {}
    for task in _unique_tasks(pipeline):
        lineage = lineage_hash(task, memo)
        new_id = f"{prefix}-{lineage[:LINEAGE_ID_LEN]}"
        n = 2
        while new_id in used_task_ids:
            logger.warning(
                f"Task ID {new_id} already used in run {run_id}; suffixing -{n}"
            )
            new_id = f"{prefix}-{lineage[:LINEAGE_ID_LEN]}-{n}"
            n += 1
        used_task_ids[new_id] = task
        task.id = new_id
        task.lineage_hash = lineage
        task.pipeline_id = run_id
        task.run_id = run_id
        task.workflows = {workflow_ids.get(w, w) for w in task.workflows}
        if use_cache is not None:
            task.config["_run_use_cache"] = bool(use_cache)
