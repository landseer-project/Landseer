from pathlib import Path

from src.data.manager import DatasetManager
from src.pipeline.config_loader import normalize_dataset_token


def test_dataset_manager_supports_mnist(tmp_path: Path) -> None:
    manager = DatasetManager(tmp_path)

    assert "mnist" in manager._loaders


def test_normalize_dataset_token_accepts_mnist() -> None:
    assert normalize_dataset_token("MNIST") == "mnist"
