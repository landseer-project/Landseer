from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.data.manager as manager_module
from src.data.loaders.celeba import CelebaLoader
from src.data.manager import DatasetManager
from src.data.types import DatasetInfo


def _write_core_celeba_artifacts(output_dir: Path, train_count: int, test_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "data.npy", np.zeros((train_count, 3, 8, 8), dtype=np.float32))
    np.save(output_dir / "labels.npy", np.zeros((train_count,), dtype=np.int64))
    np.save(output_dir / "test_data.npy", np.zeros((test_count, 3, 8, 8), dtype=np.float32))
    np.save(output_dir / "test_labels.npy", np.zeros((test_count,), dtype=np.int64))


def _write_celeba_filename_artifacts(output_dir: Path, train_count: int, test_count: int) -> None:
    train_names = [f"{idx + 1:06d}.jpg" for idx in range(train_count)]
    test_names = [f"{train_count + idx + 1:06d}.jpg" for idx in range(test_count)]
    np.save(output_dir / "filenames.npy", np.asarray(train_names, dtype=str))
    np.save(output_dir / "test_filenames.npy", np.asarray(test_names, dtype=str))


def test_celeba_loader_requires_filename_arrays(tmp_path: Path):
    output_dir = tmp_path / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=4, test_count=2)

    loader = CelebaLoader()
    with pytest.raises(FileNotFoundError, match="filenames.npy"):
        loader.prepare(output_dir=output_dir, variant="clean", params={})


def test_celeba_loader_returns_dataset_info_with_complete_artifacts(tmp_path: Path):
    output_dir = tmp_path / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=3, test_count=2)
    _write_celeba_filename_artifacts(output_dir, train_count=3, test_count=2)

    loader = CelebaLoader()
    info = loader.prepare(output_dir=output_dir, variant="clean", params={})

    assert info.name == "celeba"
    assert info.train_samples == 3
    assert info.test_samples == 2


def test_dataset_manager_uses_default_full_celeba_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("LANDSEER_CELEBA_DATA_PATH", raising=False)
    expected_path = tmp_path / "full_celeba"
    expected_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(manager_module, "DEFAULT_CELEBA_DATA_PATH", str(expected_path))

    manager = DatasetManager(tmp_path / "data")

    assert manager._resolve_output_dir("celeba", "clean") == expected_path


def test_dataset_manager_enforces_legacy_celeba_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANDSEER_CELEBA_DATA_PATH", str(tmp_path / "celeba"))

    base_dir = tmp_path / "data"
    output_dir = base_dir / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=3, test_count=1)

    class MinimalCelebaLoader:
        REQUIRED = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")

        def prepare(self, output_dir: Path, variant: str, params):
            return DatasetInfo(
                name="celeba",
                variant=variant,
                output_dir=str(output_dir),
                train_samples=3,
                test_samples=1,
            )

    manager = DatasetManager(base_dir)
    manager._loaders["celeba"] = MinimalCelebaLoader()
    manager._resolve_dataset_runtime_config = lambda key, variant: {
        "image": None,
        "labels": {},
        "source_artifacts_dir": None,
        "min_total_samples": 0,
        "require_image_dir": False,
    }

    with pytest.raises(FileNotFoundError, match="filenames.npy"):
        manager.prepare_dataset(name="celeba", variant="clean")
