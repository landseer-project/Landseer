"""The during-training step, with no defense applied (that's what "noop" means).

This just trains the model normally so we have a baseline to compare the real
defenses against. We made a CPU version because the existing in_noop image is
on ghcr and needs a token, and we wanted the whole thing runnable locally.

Folders it uses (same contract as the other tools):
  /data    dataset .npy files from the dataset stage
  /config  config_model.py with a config() function
  /output  where model.pt goes
"""

import argparse
import importlib.util
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ARTIFACTS = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")


def find_config_model(config_dir: Path) -> Path:
    """Find config_model.py.

    The Landseer worker mounts the model script at /app/config_model.py (see
    src/worker/runner.py, it mounts to /app so containers can just import it).
    Our standalone run script mounts a /config dir instead. Check both so the
    same image works either way.
    """
    for candidate in (Path("/app/config_model.py"), config_dir / "config_model.py"):
        if candidate.exists():
            return candidate
    # The worker mounts the script under its original name from the pipeline
    # yaml, e.g. /app/config_model_lenet.py, so fall back to globbing for it.
    for directory in (Path("/app"), config_dir):
        if directory.is_dir():
            matches = sorted(directory.glob("config_model*.py"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"no config_model*.py found in /app or {config_dir}"
    )


def load_model(config_dir: Path) -> nn.Module:
    """Import config_model.py at runtime and call config() to get the model."""
    cm = find_config_model(config_dir)
    print(f"[in_noop_lenet] using model config {cm}")
    # loading by file path because the file is mounted in, not installed
    spec = importlib.util.spec_from_file_location("config_model", cm)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config()


def main() -> None:
    p = argparse.ArgumentParser(description="in_noop LeNet-5 baseline trainer (CPU)")
    p.add_argument("--data", default=os.environ.get("INPUT_DIR", os.environ.get("DATA_DIR", "/data")))
    p.add_argument("--config", default=os.environ.get("CONFIG_DIR", "/config"))
    p.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    # env vars so the run script can change these without editing the image
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "3")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "128")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-3")))
    args = p.parse_args()

    data_dir, config_dir, out_dir = Path(args.data), Path(args.config), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # falls back to cpu on its own, which is what we want here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[in_noop_lenet] device={device}")

    x = np.load(data_dir / "data.npy")
    y = np.load(data_dir / "labels.npy")
    print(f"[in_noop_lenet] train data {x.shape} {x.dtype}, labels {y.shape}")

    model = load_model(config_dir).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    # handy for checking we actually got LeNet-5 (about 61.7k params) and not something else
    print(f"[in_noop_lenet] model={type(model).__name__} params={n_params:,}")

    loader = DataLoader(
        TensorDataset(torch.tensor(x).float(), torch.tensor(y).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        running, seen = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = lossf(model(bx), by)
            loss.backward()
            opt.step()
            # weighting by batch size so the average is right even on a short last batch
            running += loss.item() * by.size(0)
            seen += by.size(0)
        print(f"[in_noop_lenet] epoch {epoch+1}/{args.epochs} mean_loss={running/seen:.4f} ({time.time()-t0:.1f}s)")

    # saving the state_dict, not the whole model, so model_loader rebuilds it from config_model.py
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[in_noop_lenet] wrote {out_dir/'model.pt'} in {time.time()-t0:.1f}s")

    # copy the dataset and config along too, so /output can be fed straight
    # into the next stage without hunting for the original files
    for name in ARTIFACTS:
        src = data_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)
    shutil.copy(find_config_model(config_dir), out_dir / "config_model.py")
    print(f"[in_noop_lenet] output dir: {sorted(q.name for q in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
