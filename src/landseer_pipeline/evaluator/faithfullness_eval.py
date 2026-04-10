import numpy as np
import json
from pathlib import Path
from typing import Optional


def evaluate_faithfulness_drop10(
    drop10: np.ndarray,
    save_dir: Optional[str] = None,
    save_json: bool = True
) -> float:
    """
    Evaluate explanation faithfulness using Drop@10%.

    Args:
        drop10 (np.ndarray):
            Drop@10% scores (higher is better).
            Shape: (N,)

        save_dir (str, optional):
            Directory to save a summary JSON.
            If None, nothing is saved.

        save_json (bool):
            Whether to write faithfulness_summary.json.

    Returns:
        mean_drop10 (float):
            Mean Drop@10 score across samples.
    """
    print(f"Evaluating faithfulness drop10 with shape: {drop10.shape}")

    if drop10.ndim != 1:
        raise ValueError("drop10 must be a 1D array")

    mean_drop10 = float(drop10.mean())

    if save_dir is not None and save_json:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = {
            "n": int(len(drop10)),
            "mean_drop10": mean_drop10,
            "interpretation": "higher is better (model confidence drops when important features are removed)"
        }

        with open(out / "faithfulness_drop10_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return mean_drop10