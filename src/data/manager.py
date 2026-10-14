from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from ..common import get_logger
from ..common import dataset_manifest as dsm
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
        prepared_now = False

        # Containers are the only place that create/modify dataset artifacts.
        if missing or force_reprepare:
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
            prepared_now = True


        # # Containers are the only place that create/modify dataset artifacts.
        # if variant == "clean" and (missing or force_reprepare):
        #     if not runtime_cfg.get("image"):
        #         raise FileNotFoundError(
        #             f"Dataset artifacts missing for {key}/{variant} in {output_dir}: "
        #             f"files={missing}, dirs={missing_dirs}"
        #         )
        #     self._prepare_with_container(
        #         key=key,
        #         output_dir=output_dir,
        #         variant=variant,
        #         params=params,
        #         runtime_cfg=runtime_cfg,
        #         force_reprepare=force_reprepare,
        #     )


        info = self._build_dataset_info(
            key=key,
            variant=variant,
            output_dir=output_dir,
            required_files=required_files,
            required_dirs=required_dirs,
        )
        info.poisoning = poisoning
        self._attach_manifest(
            info,
            output_dir=output_dir,
            runtime_cfg=runtime_cfg,
            params=params,
            poisoning=poisoning,
            prepared_now=prepared_now,
        )

        meta = {
            "name": info.name,
            "variant": info.variant,
            "output_dir": info.output_dir,
            "train_samples": info.train_samples,
            "test_samples": info.test_samples,
            "poisoning": info.poisoning,
            "params": params,
            "content_id": info.content_id,
            "provenance_id": (info.provenance or {}).get("provenance_id"),
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

    # ------------------------------------------------------------------
    # Manifest + provenance
    # ------------------------------------------------------------------

    def _attach_manifest(
        self,
        info: DatasetInfo,
        output_dir: Path,
        runtime_cfg: Dict[str, Any],
        params: Dict[str, Any],
        poisoning: Optional[Dict[str, Any]],
        prepared_now: bool,
    ) -> Dict[str, Any]:
        """
        Make sure output_dir has an up-to-date dataset_manifest.json and copy its
        summary onto ``info``. Unchanged files are not rehashed (sidecar), so
        this is a stat walk after the first time.
        """
        existing = dsm.load_manifest(output_dir)
        recipe = self._recipe(info.name, info.variant, runtime_cfg, params, poisoning)
        manifest = dsm.build_manifest(
            output_dir,
            required_files=runtime_cfg.get("required_files"),
            required_dirs=runtime_cfg.get("required_dirs"),
        )
        same_content = bool(existing) and existing.get("content_id") == manifest["content_id"]

        if prepared_now:
            provenance = self._fresh_provenance(recipe, runtime_cfg)
        elif same_content and existing.get("provenance"):
            provenance = existing["provenance"]
        else:
            # Prepared before manifests existed (or edited outside the prep
            # container): record what the config says, null what we cannot know.
            provenance = dict(recipe)
            provenance.update({
                "origin": "backfilled",
                "image": None,
                "image_digest": None,
                "landseer_commit": None,
                "prep_host": None,
                "prepared_at": None,
            })
            if existing and not same_content:
                provenance["note"] = (
                    f"content changed since manifest {str(existing.get('content_id'))[:12]} "
                    "without a recorded prep run"
                )
        provenance = dsm.with_provenance_id(
            {k: v for k, v in provenance.items() if k != "provenance_id"}
        )
        manifest["provenance"] = provenance
        if same_content and not prepared_now and existing.get("manifest_created_at"):
            manifest["manifest_created_at"] = existing["manifest_created_at"]
        else:
            manifest["manifest_created_at"] = datetime.now(timezone.utc).isoformat()

        if not same_content or prepared_now or existing.get("provenance") != provenance:
            try:
                dsm.write_manifest(output_dir, manifest)
                logger.info(
                    f"Wrote dataset manifest for {info.name}/{info.variant}: "
                    f"content_id={manifest['content_id'][:12]}, {manifest['file_count']} files"
                )
            except OSError as e:
                logger.warning(f"Could not write dataset manifest in {output_dir}: {e}")

        # Same files but the config asks for a different recipe: prep only runs
        # when files are missing, so the data on disk was not regenerated.
        keys = ("params", "poisoning", "source")
        recorded = {k: provenance.get(k) for k in keys}
        wanted = {k: recipe.get(k) for k in keys}
        if provenance.get("origin") != "backfilled" and recorded != wanted:
            logger.warning(
                f"Dataset {info.name}/{info.variant} on disk was prepared with "
                f"{recorded}, but the pipeline asks for {wanted}; delete {output_dir} to re-prepare"
            )

        summary = dsm.summarize(manifest)
        info.content_id = summary["content_id"]
        info.provenance = provenance
        info.file_count = summary["file_count"]
        info.total_bytes = summary["total_bytes"]
        info.manifest_path = str(output_dir / dsm.MANIFEST_NAME)
        return manifest

    def _recipe(
        self,
        key: str,
        variant: str,
        runtime_cfg: Dict[str, Any],
        params: Dict[str, Any],
        poisoning: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Provenance fields known from config alone."""
        dataset_cfg = self._dataset_registry.get(key, {})
        variant_cfg = dataset_cfg.get("variants", {}).get(variant, {})
        return {
            "name": key,
            "variant": variant,
            "version": variant_cfg.get("version") or dataset_cfg.get("version"),
            "configured_image": runtime_cfg.get("image") or None,
            "params": params or {},
            "poisoning": poisoning,
            "source": {
                "artifacts_dir": runtime_cfg.get("source_artifacts_dir"),
                "url": variant_cfg.get("source_url") or dataset_cfg.get("source_url"),
            },
        }

    def _fresh_provenance(self, recipe: Dict[str, Any], runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
        image = runtime_cfg.get("image") or None
        prov = dict(recipe)
        prov.update({
            "origin": "prepared",
            "image": image,
            "image_digest": self._image_digest(image) if image else None,
            "landseer_commit": self._landseer_commit(),
            "prep_host": socket.gethostname(),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        })
        return prov

    @staticmethod
    def _image_digest(image: str) -> Optional[str]:
        """Repo digest (or image id) of the local image, None if docker cannot tell."""
        fmt = "{{json .RepoDigests}}|{{.Id}}"
        try:
            out = subprocess.run(
                ["docker", "image", "inspect", "--format", fmt, image],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip()
        except Exception:
            return None
        digests, _, image_id = out.partition("|")
        try:
            repo_digests = json.loads(digests) or []
        except ValueError:
            repo_digests = []
        return repo_digests[0] if repo_digests else (image_id or None)

    @staticmethod
    def _landseer_commit() -> Optional[str]:
        """HEAD of this checkout, with '-dirty' when tracked files are modified."""
        repo = Path(__file__).resolve().parents[2]
        try:
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except Exception:
            return None
        return f"{head}-dirty" if dirty else head

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
