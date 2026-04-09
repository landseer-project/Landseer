"""
Pipeline config discovery utilities.

This module scans config files from the workspace and synchronizes them with
`pipeline_configs` in the database.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

from ..common import get_logger
from ..db import PipelineConfigRepository, PipelineConfigModel, session_scope

logger = get_logger(__name__)


def _resolve_dir(path_value: str) -> Path:
    """
    Resolve config directories robustly across different working directories.

    Resolution order for relative paths:
    1. Current working directory
    2. LANDSEER_ROOT (if set)
    3. Repository root inferred from this file location
    """
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    env_root = os.getenv("LANDSEER_ROOT")
    if env_root:
        env_candidate = (Path(env_root) / path).resolve()
        if env_candidate.exists():
            return env_candidate

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config_path: str, attack_config_path: Optional[str] = None) -> str:
    """
    Compute content hash for one pipeline config (+ optional attack config).
    """
    pipeline_file = Path(config_path)
    if not pipeline_file.exists():
        return ""

    hasher = hashlib.sha256()
    hasher.update(_hash_file(pipeline_file).encode())

    if attack_config_path:
        attack_file = Path(attack_config_path)
        if attack_file.exists():
            hasher.update(_hash_file(attack_file).encode())

    return hasher.hexdigest()


def discover_configs(
    pipeline_dir: str = "configs/pipeline",
    attack_dir: str = "configs/attack",
) -> List[Dict[str, Optional[str]]]:
    """
    Discover pipeline configs from the filesystem.
    """
    configs: List[Dict[str, Optional[str]]] = []
    pdir = _resolve_dir(pipeline_dir)
    _ = _resolve_dir(attack_dir)  # reserved for future pairing logic

    if not pdir.exists():
        logger.warning("Pipeline config directory not found: %s", pdir)
        return configs

    for file in sorted(pdir.glob("*.yaml")):
        config_id = f"config_{file.stem}"
        config_hash = compute_config_hash(str(file))
        configs.append(
            {
                "id": config_id,
                "name": file.stem,
                "description": f"Discovered from {file.as_posix()}",
                "config_path": str(file.resolve()),
                "attack_config_path": None,
                "config_hash": config_hash,
            }
        )

    return configs


def sync_configs_to_db(
    pipeline_dir: str = "configs/pipeline",
    attack_dir: str = "configs/attack",
) -> List[PipelineConfigModel]:
    """
    Scan configs and upsert them into `pipeline_configs`.
    """
    discovered = discover_configs(pipeline_dir=pipeline_dir, attack_dir=attack_dir)
    synced: List[PipelineConfigModel] = []

    with session_scope() as session:
        repo = PipelineConfigRepository(session)
        for cfg in discovered:
            existing = repo.get_by_id(cfg["id"])
            if existing is None:
                synced.append(repo.create(cfg))
            else:
                synced.append(
                    repo.update(
                        cfg["id"],
                        {
                            "name": cfg["name"],
                            "description": cfg["description"],
                            "config_path": cfg["config_path"],
                            "attack_config_path": cfg["attack_config_path"],
                            "config_hash": cfg["config_hash"],
                        },
                    )
                )

    return [c for c in synced if c is not None]


def get_all_configs() -> List[PipelineConfigModel]:
    with session_scope() as session:
        return PipelineConfigRepository(session).get_all()


def get_config_by_id(config_id: str) -> Optional[PipelineConfigModel]:
    with session_scope() as session:
        return PipelineConfigRepository(session).get_by_id(config_id)

