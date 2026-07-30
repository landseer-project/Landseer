from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..common import get_logger
from .loaders import CelebaLoader, Cifar10Loader, DatasetLoader, MNISTLoader
from .types import DatasetInfo

logger = get_logger(__name__)


def _normalize_dataset_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


class DatasetManager:
    """Prepare datasets and expose a stable artifact directory contract."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._loaders: Dict[str, DatasetLoader] = {
            "cifar10": Cifar10Loader(),
            "celeba": CelebaLoader(),
            "mnist": MNISTLoader(),
        }
        self._container_images: Dict[str, str] = {
            "cifar10": os.getenv(
                "LANDSEER_DATASET_IMAGE_CIFAR10", "ghcr.io/landseer-project/dataset_cifar10:v1"
            ),
            "celeba": os.getenv(
                "LANDSEER_DATASET_IMAGE_CELEBA", "ghcr.io/landseer-project/dataset_celeba:v1"
            ),
            "mnist": os.getenv(
                "LANDSEER_DATASET_IMAGE_MNIST", "" #TODO: Add a container image for MNIST dataset preparation if needed 
            ),
        }
        self._dataset_registry = self._load_dataset_registry()

    def register_loader(self, dataset_name: str, loader: DatasetLoader) -> None:
        self._loaders[_normalize_dataset_name(dataset_name)] = loader

    def _resolve_output_dir(self, key: str, variant: str) -> Path:
        """
        Resolve dataset output dir with dataset-specific defaults.
        """
        if key in {"cifar10", "cifar100", "celeba", "mnist"}:
            variant_path = self.base_dir / key / variant
            return variant_path

    def prepare_dataset(
        self,
        name: str,
        variant: str = "clean",
        poisoning: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> DatasetInfo:
        key = _normalize_dataset_name(name)
        if key not in self._loaders:
            raise ValueError(
                f"Unsupported dataset '{name}'. Available datasets: {sorted(self._loaders.keys())}"
            )

        output_dir = self._resolve_output_dir(key=key, variant=variant)
        dataset_root = output_dir.parent
        if not output_dir.exists():
            legacy_files = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")
            if any((dataset_root / fname).exists() for fname in legacy_files):
                output_dir = dataset_root

        loader = self._loaders[key]

        required_files = tuple(getattr(loader, "REQUIRED", ()))
        missing = [name for name in required_files if not (output_dir / name).exists()]
        runtime_cfg = self._resolve_dataset_runtime_config(key=key, variant=variant)
        force_reprepare = False
        require_image_dir = bool(runtime_cfg.get("require_image_dir"))
        if key == "celeba" and variant == "clean" and require_image_dir:
            image_dir = output_dir / "img_align_celeba"
            if not image_dir.exists():
                force_reprepare = True
        if variant == "clean" and (missing or force_reprepare):
            self._prepare_with_container(
                key=key,
                output_dir=output_dir,
                variant=variant,
                params=params,
                runtime_cfg=runtime_cfg,
                force_reprepare=force_reprepare,
            )

        info = loader.prepare(output_dir=output_dir, variant=variant, params=params)

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
                f"Dataset artifacts missing for {key}/{variant} in {output_dir} and no container image configured."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        target_output_dir = output_dir
        temp_output_dir: Optional[Path] = None
        if force_reprepare:
            temp_output_dir = output_dir.parent / f"{output_dir.name}__reprepare_tmp"
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            temp_output_dir.mkdir(parents=True, exist_ok=True)
            target_output_dir = temp_output_dir
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{target_output_dir}:/output:rw",
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
        source_artifacts_dir = runtime_cfg.get("source_artifacts_dir")
        if source_artifacts_dir:
            source_dir = Path(str(source_artifacts_dir)).expanduser()
            if not source_dir.is_absolute():
                source_dir = source_dir.resolve()
            if source_dir.exists():
                cmd[3:3] = ["-v", f"{source_dir}:/source_artifacts:ro"]

        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if temp_output_dir is not None:
                required = list(getattr(self._loaders[key], "REQUIRED", ()))
                has_required = all((temp_output_dir / name).exists() for name in required)
                has_image_dir = (not runtime_cfg.get("require_image_dir")) or (temp_output_dir / "img_align_celeba").exists()
                if not has_required or not has_image_dir:
                    raise RuntimeError(
                        f"Forced reprepare for {key}/{variant} produced incomplete artifacts "
                        f"(required_files={has_required}, image_dir={has_image_dir})."
                    )
                for child in output_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                for child in temp_output_dir.iterdir():
                    dest = output_dir / child.name
                    shutil.move(str(child), str(dest))
                shutil.rmtree(temp_output_dir, ignore_errors=True)
        except subprocess.CalledProcessError as exc:
            if temp_output_dir is not None and temp_output_dir.exists():
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            raise RuntimeError(
                f"Dataset preparation container failed for {key}/{variant} using image {image}"
            ) from exc

    def _resolve_dataset_runtime_config(self, key: str, variant: str) -> Dict[str, Any]:
        dataset_cfg = self._dataset_registry.get(key, {})
        variant_cfg = dataset_cfg.get("variants", {}).get(variant, {})
        image = variant_cfg.get("image") or dataset_cfg.get("default_image") or self._container_images.get(key)
        labels = variant_cfg.get("labels") or dataset_cfg.get("labels") or {}
        source_artifacts_dir = (
            variant_cfg.get("source_artifacts_dir")
            or dataset_cfg.get("source_artifacts_dir")
            or os.getenv(f"LANDSEER_DATASET_SOURCE_{key.upper()}")
        )

        return {
            "image": image,
            "labels": labels,
            "source_artifacts_dir": source_artifacts_dir,
            "min_total_samples": variant_cfg.get("min_total_samples")
            or dataset_cfg.get("min_total_samples")
            or 0,
            "require_image_dir": variant_cfg.get("require_image_dir")
            or dataset_cfg.get("require_image_dir")
            or False,
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
                normalized = {_normalize_dataset_name(name): cfg for name, cfg in datasets.items()}
                return normalized
        except Exception:
            pass
        return {}
