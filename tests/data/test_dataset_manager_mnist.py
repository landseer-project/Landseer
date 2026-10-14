from pathlib import Path

import pytest
import yaml

from src.data.manager import DatasetManager
from src.pipeline.config_loader import normalize_dataset_token


def test_dataset_manager_supports_mnist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "mnist": {
                        "required_files": [
                            "data.npy",
                            "labels.npy",
                            "test_data.npy",
                            "test_labels.npy",
                        ],
                        "variants": {"clean": {"image": ""}},
                    }
                }
            }
        )
    )
    monkeypatch.setenv("LANDSEER_DATASETS_CONFIG", str(cfg))

    manager = DatasetManager(tmp_path)
    assert "mnist" in manager.list_datasets()


def test_normalize_dataset_token_accepts_mnist() -> None:
    assert normalize_dataset_token("MNIST") == "mnist"
