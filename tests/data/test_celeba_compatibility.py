from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

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


def _write_datasets_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "celeba": {
                        "required_files": [
                            "data.npy",
                            "labels.npy",
                            "test_data.npy",
                            "test_labels.npy",
                            "filenames.npy",
                            "test_filenames.npy",
                        ],
                        "required_dirs": ["img_align_celeba"],
                        "variants": {"clean": {"image": ""}},
                    },
                    "mnist": {
                        "required_files": [
                            "data.npy",
                            "labels.npy",
                            "test_data.npy",
                            "test_labels.npy",
                        ],
                        "variants": {"clean": {"image": ""}},
                    },
                }
            }
        )
    )


def test_celeba_requires_filename_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "datasets.yaml"
    _write_datasets_config(cfg)
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(cfg))

    base_dir = tmp_path / "data"
    output_dir = base_dir / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=4, test_count=2)
    (output_dir / "img_align_celeba").mkdir()

    manager = DatasetManager(base_dir)
    with pytest.raises(FileNotFoundError, match="filenames.npy"):
        manager.prepare_dataset(name="celeba", variant="clean")


def test_celeba_returns_dataset_info_with_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "datasets.yaml"
    _write_datasets_config(cfg)
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(cfg))

    base_dir = tmp_path / "data"
    output_dir = base_dir / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=3, test_count=2)
    _write_celeba_filename_artifacts(output_dir, train_count=3, test_count=2)
    (output_dir / "img_align_celeba").mkdir()

    manager = DatasetManager(base_dir)
    info = manager.prepare_dataset(name="celeba", variant="clean")

    assert isinstance(info, DatasetInfo)
    assert info.name == "celeba"
    assert info.train_samples == 3
    assert info.test_samples == 2
    assert info.output_dir == str(output_dir)


def test_dataset_manager_resolves_celeba_under_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tmp_path / "datasets.yaml"
    _write_datasets_config(cfg)
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(cfg))

    base_dir = tmp_path / "data"
    manager = DatasetManager(base_dir)
    assert manager._resolve_output_dir("celeba", "clean") == base_dir / "celeba" / "clean"


def test_dataset_manager_enforces_celeba_filename_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "datasets.yaml"
    _write_datasets_config(cfg)
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(cfg))

    base_dir = tmp_path / "data"
    output_dir = base_dir / "celeba" / "clean"
    _write_core_celeba_artifacts(output_dir, train_count=3, test_count=1)
    (output_dir / "img_align_celeba").mkdir()

    manager = DatasetManager(base_dir)
    with pytest.raises(FileNotFoundError, match="filenames.npy"):
        manager.prepare_dataset(name="celeba", variant="clean")
