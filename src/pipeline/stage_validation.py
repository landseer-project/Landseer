"""
Validate that each pipeline stage only lists tools whose declared stage matches.

Uses ``defense_stage`` / ``stage`` from ``tools.yaml`` and/or OCI image labels
(``org.opencontainers.image.stage``, ``org.opencontainers.image.defense_stage``).
There is **no** inference from tool id prefixes — explicit YAML or image labels only.

When no stage information is available for a tool, validation is skipped for that
tool (same as legacy when labels were missing). Unknown tool ids raise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .container_labels import get_container_labels_for_image
from .tools import ToolDefinition, load_tools_from_yaml

logger = logging.getLogger(__name__)

REQUIRED_PIPELINE_STAGES = (
    "pre_training",
    "during_training",
    "post_training",
    "deployment",
)

# Legacy landseer_pipeline/config/schemas/pipeline.py
STAGE_SYNONYMS: Dict[str, frozenset] = {
    "pre_training": frozenset(
        {"pre_training", "pre", "pretrain", "pre_defense"},
    ),
    "during_training": frozenset(
        {
            "during_training",
            "during",
            "in",
            "train",
            "training",
            "in_training",
            "in_defense",
            "during_defense",
        },
    ),
    "post_training": frozenset(
        {"post_training", "post", "after", "posttrain", "post_defense"},
    ),
    "deployment": frozenset(
        {"deployment", "deploy", "inference", "deploy_defense"},
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_tools_yaml_path(yaml_path: str) -> Path:
    p = Path(yaml_path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return _repo_root() / p


def _effective_stage_label(
    tool_def: ToolDefinition,
    labels: Dict[str, str],
) -> Optional[str]:
    if tool_def.defense_stage:
        return tool_def.defense_stage.strip().lower()
    raw = labels.get("org.opencontainers.image.stage") or labels.get(
        "org.opencontainers.image.defense_stage"
    )
    if raw:
        return str(raw).strip().lower()
    return None


def validate_pipeline_tool_stages(
    pipeline: Dict[str, object],
    tools_by_id: Dict[str, ToolDefinition],
    *,
    fetch_remote_labels: bool = True,
) -> None:
    """
    For each tool under a pipeline stage, ensure declared stage matches that stage.

    Raises:
        ValueError: unknown tool id, or stage mismatch when a stage label is present.
    """
    for stage_name in REQUIRED_PIPELINE_STAGES:
        stage_cfg = pipeline.get(stage_name)
        if stage_cfg is None:
            continue
        raw_tools = getattr(stage_cfg, "tools", None) or []
        tool_ids: List[str] = []
        for tool_entry in raw_tools:
            # Supports both legacy list[str] and StageToolConfig/list[dict] styles.
            if isinstance(tool_entry, str):
                tool_ids.append(tool_entry)
            else:
                tool_name = getattr(tool_entry, "tool", None)
                if tool_name:
                    tool_ids.append(str(tool_name))
        allowed = STAGE_SYNONYMS.get(stage_name, frozenset({stage_name}))

        for tool_id in tool_ids:
            tool_def = tools_by_id.get(tool_id)
            if tool_def is None:
                raise ValueError(
                    f"Unknown tool {tool_id!r} — not defined in the tools catalog "
                    f"used for this pipeline load."
                )

            labels: Dict[str, str] = {}
            if fetch_remote_labels and not tool_def.defense_stage:
                labels = get_container_labels_for_image(
                    tool_def.container.image,
                    tool_def.container.runtime,
                )

            label_token = _effective_stage_label(tool_def, labels)
            if label_token is None:
                logger.debug(
                    "No defense_stage in YAML and no stage labels for tool %r "
                    "under stage %r — skipping stage check (legacy behavior)",
                    tool_id,
                    stage_name,
                )
                continue
            if label_token not in allowed:
                raise ValueError(
                    f"Tool {tool_id!r} is listed under pipeline stage {stage_name!r}, "
                    f"but its effective stage label is {label_token!r}. "
                    f"Allowed synonyms: {sorted(allowed)}"
                )
            logger.debug(
                "Stage validation OK: tool=%r pipeline_stage=%r label=%r",
                tool_id,
                stage_name,
                label_token,
            )


def load_tools_and_validate_pipeline_stages(
    pipeline: Dict[str, object],
    tools_yaml_path: str = "configs/tools.yaml",
    *,
    fetch_remote_labels: bool = True,
) -> Dict[str, ToolDefinition]:
    path = resolve_tools_yaml_path(tools_yaml_path)
    tools_by_id = load_tools_from_yaml(str(path))
    validate_pipeline_tool_stages(
        pipeline,
        tools_by_id,
        fetch_remote_labels=fetch_remote_labels,
    )
    return tools_by_id
