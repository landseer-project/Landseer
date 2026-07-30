from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..common import get_logger
from .types import DatasetInfo

logger = get_logger(__name__)

_CORE_REQUIRED = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


class DatasetManager:
    """Prepare datasets via containers and expose a stable artifact directory contract."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._container_image_env: Dict[str, str] = {
            "cifar10": os.getenv(
                "LANDSEER_DATASET_IMAGE_CIFAR10", "ghcr.io/landseer-project/dataset_cifar10:v1"
            ),
            "celeba": os.getenv(
                "LANDSEER_DATASET_IMAGE_CELEBA", "ghcr.io/landseer-project/dataset_celeba:v1"
            ),
            "mnist": os.getenv("LANDSEER_DATASET_IMAGE_MNIST", ""),
        }
        self._dataset_registry = self._load_dataset_registry()

    def list_datasets(self) -> List[str]:
        return sorted(self._dataset_registry.keys())

    def _resolve_output_dir(self, key: str, variant: str) -> Path:
        return self.base_dir / key / variant

    def prepare_dataset(
        self,
        name: str,
        variant: str = "clean",
        poisoning: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> DatasetInfo:
        key = _normalize_dataset_name(name)
        if key not in self._dataset_registry:
            raise ValueError(
                f"Unsupported dataset '{name}'. Available datasets: {self.list_datasets()}"
            )

        output_dir = self._resolve_output_dir(key=key, variant=variant)
        dataset_root = output_dir.parent
        if not output_dir.exists():
            if any((dataset_root / fname).exists() for fname in _CORE_REQUIRED):
                output_dir = dataset_root

        runtime_cfg = self._resolve_dataset_runtime_config(key=key, variant=variant)
        required_files = list(runtime_cfg["required_files"])
        required_dirs = list(runtime_cfg["required_dirs"])

        missing = [fname for fname in required_files if not (output_dir / fname).exists()]
        missing_dirs = [dname for dname in required_dirs if not (output_dir / dname).is_dir()]
        force_reprepare = bool(missing_dirs)

        # Containers are the only place that create/modify dataset artifacts.
        if variant == "clean" and (missing or force_reprepare):
            if not runtime_cfg.get("image"):
                raise FileNotFoundError(
                    f"Dataset artifacts missing for {key}/{variant} in {output_dir}: "
                    f"files={missing}, dirs={missing_dirs}"
                )
            self._prepare_with_container(
                key=key,
                output_dir=output_dir,
                variant=variant,
                params=params,
                runtime_cfg=runtime_cfg,
                force_reprepare=force_reprepare,
            )

        info = self._build_dataset_info(
            key=key,
            variant=variant,
            output_dir=output_dir,
            required_files=required_files,
            required_dirs=required_dirs,
        )
        info.poisoning = poisoning

        meta = {
            "name": info.name,
            "variant": info.variant,
            "output_dir": info.output_dir,
            "train_samples": info.train_samples,
            "test_samples": info.test_samples,
            "poisoning": info.poisoning,
            "params": params,
        }
        dataset_root.mkdir(parents=True, exist_ok=True)
        (dataset_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=str))

        return info

    def _build_dataset_info(
        self,
        key: str,
        variant: str,
        output_dir: Path,
        required_files: List[str],
        required_dirs: List[str],
    ) -> DatasetInfo:
        missing = [fname for fname in required_files if not (output_dir / fname).exists()]
        missing_dirs = [dname for dname in required_dirs if not (output_dir / dname).is_dir()]
        if missing or missing_dirs:
            raise FileNotFoundError(
                f"Dataset '{key}' is missing required artifacts in {output_dir}: "
                f"files={missing}, dirs={missing_dirs}"
            )

        train_samples = int(np.load(output_dir / "labels.npy").shape[0])
        test_samples = int(np.load(output_dir / "test_labels.npy").shape[0])
        return DatasetInfo(
            name=key,
            variant=variant,
            output_dir=str(output_dir),
            train_samples=train_samples,
            test_samples=test_samples,
        )

    def _prepare_with_container(
        self,
        key: str,
        output_dir: Path,
        variant: str,
        params: Dict[str, Any],
        runtime_cfg: Optional[Dict[str, Any]] = None,
        force_reprepare: bool = False,
    ) -> None:
        runtime_cfg = runtime_cfg or self._resolve_dataset_runtime_config(key=key, variant=variant)
        image = runtime_cfg.get("image")
        if not image:
            raise FileNotFoundError(
                f"Dataset artifacts missing for {key}/{variant} in {output_dir} "
                "and no container image configured."
            )

        # Write directly into output_dir. Run as the host user so created files
        # are owned by us (no root-owned temp dirs to clean up later).
        if force_reprepare and output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["docker", "run", "--rm"]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

        source_artifacts_dir = runtime_cfg.get("source_artifacts_dir")
        if source_artifacts_dir:
            source_dir = Path(str(source_artifacts_dir)).expanduser()
            if not source_dir.is_absolute():
                source_dir = source_dir.resolve()
            if source_dir.exists():
                cmd += ["-v", f"{source_dir}:/source_artifacts:ro"]

        cmd += [
            "-v",
            f"{output_dir}:/output:rw",
            "-e",
            f"DATASET_NAME={key}",
            "-e",
            f"DATASET_VARIANT={variant}",
            "-e",
            f"DATASET_PARAMS={json.dumps(params, sort_keys=True)}",
            "-e",
            f"LANDSEER_DATASET_LABELS={json.dumps(runtime_cfg.get('labels', {}), sort_keys=True)}",
            image,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Dataset preparation container failed for {key}/{variant} using image {image}"
            ) from exc

    @staticmethod
    def _first_defined(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _resolve_dataset_runtime_config(self, key: str, variant: str) -> Dict[str, Any]:
        dataset_cfg = self._dataset_registry.get(key, {})
        variant_cfg = dataset_cfg.get("variants", {}).get(variant, {})
        # Explicit empty string means "no image" (do not fall through to env defaults).
        image = self._first_defined(
            variant_cfg["image"] if "image" in variant_cfg else None,
            dataset_cfg["default_image"] if "default_image" in dataset_cfg else None,
            self._container_image_env.get(key),
            "",
        )
        labels = variant_cfg.get("labels") or dataset_cfg.get("labels") or {}
        source_artifacts_dir = (
            variant_cfg.get("source_artifacts_dir")
            or dataset_cfg.get("source_artifacts_dir")
            or os.getenv(f"LANDSEER_DATASET_SOURCE_{key.upper()}")
        )

        required_files = (
            variant_cfg.get("required_files")
            or dataset_cfg.get("required_files")
            or list(_CORE_REQUIRED)
        )
        required_dirs = list(
            variant_cfg.get("required_dirs") or dataset_cfg.get("required_dirs") or []
        )

        return {
            "image": image,
            "labels": labels,
            "source_artifacts_dir": source_artifacts_dir,
            "required_files": list(required_files),
            "required_dirs": required_dirs,
            "require_image_dir": bool(required_dirs),
        }

    def _load_dataset_registry(self) -> Dict[str, Any]:
        cfg_path = os.getenv("LANDSEER_DATASETS_CONFIG", "configs/datasets.yaml")
        path = Path(cfg_path)
        if not path.exists():
            return {}
        try:
            raw = yaml.safe_load(path.read_text()) or {}
            datasets = raw.get("datasets", {})
            if isinstance(datasets, dict):
                return {_normalize_dataset_name(name): cfg for name, cfg in datasets.items()}
        except Exception:
            logger.exception("Failed to load dataset registry from %s", path)
        return {}
