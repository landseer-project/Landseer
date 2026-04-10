# watermark_eval.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Literal


class NPYDatasetXY(Dataset):
    def __init__(self, data_path: str, labels_path: str):
        self.X = np.load(data_path).astype(np.float32)
        self.y = np.load(labels_path).astype(np.int64)

        if self.X.ndim != 4:
            raise ValueError(f"Expected X.ndim==4, got {self.X.ndim} from {data_path}")
        if self.y.ndim != 1:
            raise ValueError(f"Expected y.ndim==1, got {self.y.ndim} from {labels_path}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"N mismatch: X has {self.X.shape[0]}, y has {self.y.shape[0]}")

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), int(self.y[idx])


def _norm_fn(
    device: torch.device,
    mode: Literal["none", "half", "cifar10"] = "cifar10",
):
    """
    mode:
      - "none": no normalization
      - "half": (x-0.5)/0.5   (maps [0,1] -> [-1,1])
      - "cifar10": (x-mean)/std using CIFAR-10 stats
    """
    if mode == "none":
        return lambda x: x

    if mode == "half":
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        return lambda x: (x - mean) / std

    if mode == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
        return lambda x: (x - mean) / std

    raise ValueError(f"Unknown norm mode: {mode}")


@torch.no_grad()
def _eval_acc(model: torch.nn.Module, loader: DataLoader, device: torch.device, norm_mode: str) -> float:
    model.eval()
    norm = _norm_fn(device, norm_mode)
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(norm(xb))
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(total, 1)


def evaluate_wmacc_from_paths(
    *,
    model_ctor: Callable[[], torch.nn.Module],
    model_ckpt: str,
    wm_test_data: str = "data/wm_test_data.npy",
    wm_test_labels: str = "data/wm_test_labels.npy",
    batch_size: int = 128,
    num_workers: int = 2,
    device: Optional[str] = None,
    norm_mode: Literal["none", "half", "cifar10"] = "cifar10",
) -> float:
    """
    Returns watermark accuracy (%) on wm_test set.
    You can call this inside any pipeline after training.

    model_ctor: function that returns the model architecture (e.g., config from config_model.py)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model_ctor().to(device)
    sd = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(sd)

    ds = NPYDatasetXY(wm_test_data, wm_test_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    wmacc = _eval_acc(model, loader, device, norm_mode=norm_mode)
    return float(wmacc)