from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms

from data.poisoning.strategies.badnets import BadNetsStrategy


REQUIRED = (
    "data.npy",
    "labels.npy",
    "test_data.npy",
    "test_labels.npy",
    "poisoning_metadata.json",
)


def _already_prepared(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BadNets-poisoned CIFAR-10 artifacts for Landseer")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--download-dir", default=None)

    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--poison-seed", type=int, default=42)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--trigger-value", type=float, default=None)
    parser.add_argument(
        "--trigger-position",
        default="bottom_right",
        choices=["bottom_right", "bottom_left", "top_right", "top_left", "center"],
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _already_prepared(output_dir):
        print(f"BadNets CIFAR-10 artifacts already exist in {output_dir}")
        return

    download_dir = Path(args.download_dir) if args.download_dir else output_dir / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR-10 contract: CHW float32 in [0,1]
    to_tensor = transforms.ToTensor()

    train_ds = torchvision.datasets.CIFAR10(
        root=str(download_dir),
        train=True,
        download=True,
        transform=to_tensor,
    )

    test_ds = torchvision.datasets.CIFAR10(
        root=str(download_dir),
        train=False,
        download=True,
        transform=to_tensor,
    )

    train_x = np.stack([np.asarray(img) for img, _ in train_ds]).astype(np.float32)
    train_y = np.asarray([label for _, label in train_ds], dtype=np.int64)

    test_x = np.stack([np.asarray(img) for img, _ in test_ds]).astype(np.float32)
    test_y = np.asarray([label for _, label in test_ds], dtype=np.int64)

    print(
        f"Train before poisoning: shape={train_x.shape}, "
        f"dtype={train_x.dtype}, range=[{train_x.min()}, {train_x.max()}]"
    )

    trigger_value = 1.0 if args.trigger_value is None else args.trigger_value

    poisoner = BadNetsStrategy()

    result = poisoner.apply(
        data=train_x,
        labels=train_y,
        dataset_id="cifar10",
        poison_rate=args.poison_rate,
        target_class=args.target_class,
        seed=args.poison_seed,
        trigger_size=args.trigger_size,
        trigger_value=trigger_value,
        trigger_position=args.trigger_position,
    )

    poisoned_train_x = np.asarray(result.data, dtype=np.float32)
    poisoned_train_y = np.asarray(result.labels, dtype=np.int64)

    if poisoned_train_x.shape[0] != train_x.shape[0]:
        raise ValueError(
            f"Poisoning changed train size: "
            f"{train_x.shape[0]} -> {poisoned_train_x.shape[0]}"
        )

    if poisoned_train_x.min() < 0.0 or poisoned_train_x.max() > 1.0:
        raise ValueError(
            f"Poisoned CIFAR-10 data outside [0,1]: "
            f"[{poisoned_train_x.min()}, {poisoned_train_x.max()}]"
        )

    poisoned_indices = np.flatnonzero(poisoned_train_y != train_y)
    actual_poison_rate = len(poisoned_indices) / len(poisoned_train_x)

    attack_info = {
        "technique": "badnets",
        "trigger_size": args.trigger_size,
        "trigger_value": float(trigger_value),
        "trigger_position": args.trigger_position,
    }

    poisoning_metadata = {
        "dataset_id": "cifar10",
        "dataset_variant": "badnets",
        "poison_type": "backdoor",
        "strategy": "badnets",
        "target_class": args.target_class,
        "poison_rate_requested": args.poison_rate,
        "poison_rate_actual": actual_poison_rate,
        "num_poisoned": int(len(poisoned_indices)),
        "poison_seed": args.poison_seed,
        "poisoned_indices": poisoned_indices.astype(int).tolist(),
        "attack_info": attack_info,
        "trigger_info": attack_info,
    }

    np.save(output_dir / "data.npy", poisoned_train_x)
    np.save(output_dir / "labels.npy", poisoned_train_y)
    np.save(output_dir / "test_data.npy", test_x)
    np.save(output_dir / "test_labels.npy", test_y)

    (output_dir / "poisoning_metadata.json").write_text(
        json.dumps(poisoning_metadata, indent=2)
    )

    print(f"CIFAR-10 BadNets artifacts written to {output_dir}")
    print(
        f"Train={len(poisoned_train_x)}, test={len(test_x)}, "
        f"poisoned={len(poisoned_indices)}"
    )
    print(
        f"Train dtype={poisoned_train_x.dtype}, "
        f"range=[{poisoned_train_x.min()}, {poisoned_train_x.max()}]"
    )
    print(
        f"Poison rate: requested={args.poison_rate:.4f}, "
        f"actual={actual_poison_rate:.4f}"
    )
    print(
        f"Target={args.target_class}, trigger={args.trigger_size}x{args.trigger_size}, "
        f"position={args.trigger_position}, value={trigger_value}"
    )


if __name__ == "__main__":
    main()