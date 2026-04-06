"""
Shared PyTorch checkpoint loading for Landseer evaluators.

Handles:
- Full ``nn.Module`` checkpoints (typical ``torch.save(model, ...)``).
- Raw ``state_dict`` when ``config_model.py`` is present in the evaluator input dir
  (typical ``torch.save(model.state_dict(), ...)``).
- Clear errors for empty, truncated, or non-checkpoint files (common cause of
  ``pickle.UnpicklingError: Ran out of input`` inside ``torch.load``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn


def _checkpoint_size_hint(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError as e:
        return f"stat failed: {e}"
    return f"{n} bytes"


def load_torch_model_for_eval(
    model_path: Path,
    input_dir: Path,
    device: Union[str, torch.device],
) -> nn.Module:
    """
    Load a model for evaluation.

    Uses ``weights_only=False`` so full modules and legacy checkpoints load
    correctly across PyTorch versions that default ``weights_only`` to True.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If the file is empty, not a loadable checkpoint, or a
            state_dict is present without ``config_model.py``.
    """
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    try:
        size = model_path.stat().st_size
    except OSError as e:
        raise ValueError(f"Cannot stat model file {model_path}: {e}") from e
    if size == 0:
        raise ValueError(
            "model.pt is empty (0 bytes). Upstream may have failed or merged the wrong file."
        )

    try:
        loaded: Any = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
    except Exception as e:
        hint = _checkpoint_size_hint(model_path)
        raise ValueError(
            f"torch.load failed ({hint}): {e}. "
            "Often indicates a truncated or non-PyTorch file named model.pt."
        ) from e

    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded

    if isinstance(loaded, dict):
        if "model" in loaded and isinstance(loaded["model"], nn.Module):
            m = loaded["model"]
            m.eval()
            return m
        state: dict[str, Any]
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state = loaded["state_dict"]
        elif _is_plain_state_dict(loaded):
            state = loaded
        else:
            raise ValueError(
                "Unrecognized checkpoint dict (expected state_dict tensors, "
                "or keys 'state_dict' / 'model')."
            )
        return _build_model_from_state_dict(state, input_dir, device)

    raise ValueError(
        f"Unsupported checkpoint type {type(loaded).__name__}; "
        "expected nn.Module or state_dict-like dict."
    )


def _is_plain_state_dict(d: dict[str, Any]) -> bool:
    if not d:
        return False
    if not all(isinstance(k, str) for k in d):
        return False
    vals = list(d.values())
    return bool(vals) and all(isinstance(v, torch.Tensor) for v in vals)


def _build_model_from_state_dict(
    state: dict[str, Any],
    input_dir: Path,
    device: Union[str, torch.device],
) -> nn.Module:
    """Instantiate model from ``config_model.py`` and load ``state``."""
    cm = input_dir / "config_model.py"
    if not cm.exists():
        raise ValueError(
            "Checkpoint is a state_dict but input/config_model.py is missing; "
            "cannot construct the network. Ensure the dataset/model script is merged into the task input."
        )
    spec = importlib.util.spec_from_file_location("config_model_eval", cm)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from {cm}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "config"):
        raise ValueError(f"{cm} has no config() function")
    model = mod.config()
    if not isinstance(model, nn.Module):
        raise ValueError("config() must return an nn.Module")
    try:
        model.load_state_dict(state)
    except Exception as e:
        state2 = _maybe_strip_prefix(state, "module.")
        if state2 is not state:
            model.load_state_dict(state2)
        else:
            raise ValueError(
                f"state_dict does not match model architecture: {e}"
            ) from e
    model = model.to(device)
    model.eval()
    return model


def _maybe_strip_prefix(
    state: dict[str, Any], prefix: str
) -> dict[str, Any]:
    if not all(isinstance(k, str) for k in state):
        return state
    keys = list(state.keys())
    if not keys or not all(k.startswith(prefix) for k in keys):
        return state
    return {k[len(prefix) :]: v for k, v in state.items()}
