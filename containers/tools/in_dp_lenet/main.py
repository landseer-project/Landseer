#!/usr/bin/env python3
"""Differential privacy defense (DP-SGD via Opacus), during-training stage, CPU.

We wrote this because the existing in_dp image on ghcr only works with cifar10.
It applies transforms.Normalize with 3-channel constants, so a 1-channel mnist
image makes it crash with:
    output with shape [1, 28, 28] doesn't match the broadcast shape [3, 28, 28]

This version reads whatever shape the dataset actually is and doesn't normalize
at all (the dataset container already scales to [0,1]), so it works for mnist
and would work for cifar10 too.

DP-SGD in three parts:
  1. clip each sample's gradient to max_grad_norm so one example can't dominate
  2. add gaussian noise to the summed gradient, which is what hides individuals
  3. track the privacy budget (epsilon) spent across all the training steps

Opacus can't handle BatchNorm (it mixes samples within a batch, which breaks
per-sample gradients). LeNet-5 has no normalization layers so it works as-is.

Writes privacy_metrics.txt alongside model.pt, which is how the clean evaluator
picks up epsilon and dp_accuracy (see containers/evals/clean/evaluate.py).
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

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

ARTIFACTS = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")


def find_config_model(config_dir: Path) -> Path:
    """Locate config_model.py, same lookup as in_noop_lenet.

    The landseer worker mounts the model script at /app under its original
    filename from the pipeline yaml, our standalone script uses /config.
    """
    for candidate in (Path("/app/config_model.py"), config_dir / "config_model.py"):
        if candidate.exists():
            return candidate
    for directory in (Path("/app"), config_dir):
        if directory.is_dir():
            matches = sorted(directory.glob("config_model*.py"))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"no config_model*.py found in /app or {config_dir}")


def disable_inplace(model: nn.Module) -> int:
    """Turn off in-place activations.

    Opacus attaches backward hooks to get per-sample gradients, and an in-place
    op modifies a view those hooks created, which pytorch refuses:
        "Output 0 of BackwardHookFunction is a view and is being modified inplace"
    inplace=False is the same maths, just uses a bit more memory. We fix it here
    rather than in the model config so the shared config stays untouched and any
    model handed to this tool works.
    """
    fixed = 0
    for module in model.modules():
        if getattr(module, "inplace", False):
            module.inplace = False
            fixed += 1
    return fixed


def load_model(config_dir: Path) -> nn.Module:
    cm = find_config_model(config_dir)
    print(f"[in_dp_lenet] using model config {cm}")
    spec = importlib.util.spec_from_file_location("config_model", cm)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.config()
    n = disable_inplace(model)
    if n:
        print(f"[in_dp_lenet] disabled inplace on {n} module(s) for opacus compatibility")
    return model


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="DP-SGD baseline defense (CPU)")
    p.add_argument("--data", default=os.environ.get("INPUT_DIR", os.environ.get("DATA_DIR", "/data")))
    p.add_argument("--config", default=os.environ.get("CONFIG_DIR", "/config"))
    p.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "/output"))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "5")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-3")))
    # privacy budget. lower epsilon = more private = worse accuracy.
    # 8.0 is the usual "moderate" setting people report.
    p.add_argument("--epsilon", type=float, default=float(os.environ.get("EPSILON", "8.0")))
    p.add_argument("--delta", type=float, default=float(os.environ.get("DELTA", "1e-5")))
    p.add_argument("--max-grad-norm", type=float, default=float(os.environ.get("MAX_GRAD_NORM", "1.0")))
    args = p.parse_args()

    data_dir, config_dir, out_dir = Path(args.data), Path(args.config), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[in_dp_lenet] device={device}")

    x = np.load(data_dir / "data.npy")
    y = np.load(data_dir / "labels.npy")
    xt = np.load(data_dir / "test_data.npy")
    yt = np.load(data_dir / "test_labels.npy")
    print(f"[in_dp_lenet] train {x.shape} {x.dtype}, test {xt.shape}")

    model = load_model(config_dir).to(device)
    # opacus refuses models it can't do per-sample gradients on. LeNet is fine,
    # but check anyway so the failure is a clear message instead of a crash.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"[in_dp_lenet] model not DP-compatible, fixing: {errors}")
        model = ModuleValidator.fix(model).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[in_dp_lenet] model={type(model).__name__} params={n_params:,}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x).float(), torch.tensor(y).long()),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(xt).float(), torch.tensor(yt).long()),
        batch_size=512, shuffle=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    # make_private_with_epsilon works backwards from the budget we want: given
    # epsilon, delta and how many epochs we'll run, it picks the noise level.
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, opt, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
    )
    print(f"[in_dp_lenet] target epsilon={args.epsilon} delta={args.delta} "
          f"max_grad_norm={args.max_grad_norm} noise={opt.noise_multiplier:.4f}")

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = lossf(model(bx), by)
            loss.backward()
            opt.step()
            running += loss.item() * by.size(0)
            seen += by.size(0)
        eps = privacy_engine.get_epsilon(args.delta)
        print(f"[in_dp_lenet] epoch {epoch+1}/{args.epochs} loss={running/max(seen,1):.4f} "
              f"eps_spent={eps:.3f} ({time.time()-t0:.1f}s)")

    final_eps = privacy_engine.get_epsilon(args.delta)
    test_acc = accuracy(model, test_loader, device)
    print(f"[in_dp_lenet] done in {time.time()-t0:.1f}s | test_acc={test_acc:.4f} | epsilon={final_eps:.3f}")

    # opacus wraps the model in GradSampleModule, so unwrap before saving or the
    # state_dict keys get a _module. prefix and the evaluator can't load it.
    to_save = model._module if hasattr(model, "_module") else model
    torch.save(to_save.state_dict(), out_dir / "model.pt")

    # the clean evaluator looks for this file and pulls epsilon / dp_accuracy
    # out of it, which is how they end up in metrics_summary.csv
    (out_dir / "privacy_metrics.txt").write_text(
        f"epsilon={final_eps:.6f}\n"
        f"delta={args.delta}\n"
        f"dp_accuracy={test_acc:.6f}\n"
        f"noise_multiplier={opt.noise_multiplier:.6f}\n"
        f"max_grad_norm={args.max_grad_norm}\n"
    )

    for name in ARTIFACTS:
        src = data_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)
    shutil.copy(find_config_model(config_dir), out_dir / "config_model.py")
    print(f"[in_dp_lenet] output: {sorted(q.name for q in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
