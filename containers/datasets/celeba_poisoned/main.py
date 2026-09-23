from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from data.poisoning.strategies.badnets import BadNetsStrategy


REQUIRED = (
    "data.npy",
    "labels.npy",
    "test_data.npy",
    "test_labels.npy",
    "filenames.npy",
    "test_filenames.npy",
    "poisoning_metadata.json",
)


def _already_prepared(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED) and (output_dir / "img_align_celeba").is_dir()


def _resolve_image_dir(source_root: Path) -> Path:
    candidates = [
        source_root / "img_align_celeba" / "img_align_celeba",
        source_root / "img_align_celeba",
    ]

    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate

    raise FileNotFoundError(
        "Could not find CelebA jpg images under "
        f"{source_root}/img_align_celeba[/img_align_celeba]."
    )


def _resolve_attr_csv(source_root: Path) -> Path:
    attr_path = source_root / "list_attr_celeba.csv"
    if not attr_path.exists():
        raise FileNotFoundError(f"Missing attribute CSV: {attr_path}")
    return attr_path


def _load_image_chw(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if image_size > 0:
            rgb = rgb.resize((image_size, image_size), Image.BILINEAR)
        # arr = np.asarray(rgb, dtype=np.float32)
        # Landseer CelebA contract: CHW float32 in [0, 1]
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    return np.transpose(arr, (2, 0, 1))


def _load_split(
    img_dir: Path,
    file_names: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    output_img_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    images, valid_labels, valid_names = [], [], []
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for filename, label in zip(file_names, labels):
        name = str(filename)
        src = img_dir / name

        try:
            arr = _load_image_chw(src, image_size)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue

        images.append(arr)
        valid_labels.append(int(label))
        valid_names.append(name)

        hwc = np.transpose(arr.astype(np.uint8), (1, 2, 0))
        Image.fromarray(hwc, mode="RGB").save(output_img_dir / name, quality=95)

    if not images:
        raise RuntimeError(f"No images could be loaded from {img_dir}")

    return (
        np.stack(images),
        np.asarray(valid_labels, dtype=np.int64),
        np.asarray(valid_names, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BadNets-poisoned CelebA artifacts for Landseer")

    parser.add_argument("--output", default="/output")
    parser.add_argument("--source-artifacts-dir", default="/source_artifacts")
    parser.add_argument("--target-attribute", default="Smiling")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--target-class", type=int, default=1)
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
        print(f"BadNets CelebA artifacts already exist in {output_dir}")
        return

    source_root = Path(args.source_artifacts_dir)

    if not source_root.exists():
        raise FileNotFoundError(f"Source CelebA directory not found: {source_root}")

    img_dir = _resolve_image_dir(source_root)
    attr_path = _resolve_attr_csv(source_root)

    df = pd.read_csv(attr_path)

    if "image_id" not in df.columns:
        raise ValueError(f"{attr_path} must contain an 'image_id' column")

    if args.target_attribute not in df.columns:
        raise ValueError(f"Attribute '{args.target_attribute}' not found in {attr_path}")

    filenames = df["image_id"].astype(str).values
    labels = (df[args.target_attribute].values == 1).astype(np.int64)

    if args.limit > 0:
        filenames = filenames[:args.limit]
        labels = labels[:args.limit]

    train_files, test_files, train_labels, test_labels = train_test_split(
        filenames,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    out_img_dir = output_dir / "img_align_celeba"

    if out_img_dir.exists():
        shutil.rmtree(out_img_dir)

    train_x, train_y, train_names = _load_split(
        img_dir,
        train_files,
        train_labels,
        args.image_size,
        out_img_dir,
    )

    test_x, test_y, test_names = _load_split(
        img_dir,
        test_files,
        test_labels,
        args.image_size,
        out_img_dir,
    )

    print(f"Train before poisoning: shape={train_x.shape}, range=[{train_x.min()}, {train_x.max()}]")

    poisoner = BadNetsStrategy()

    result = poisoner.apply(
        data=train_x,
        labels=train_y,
        dataset_id="celeba",
        poison_rate=args.poison_rate,
        target_class=args.target_class,
        seed=args.poison_seed,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_position=args.trigger_position,
    )

    poisoned_train_x = result.data
    poisoned_train_y = result.labels

    poisoned_indices = np.flatnonzero(poisoned_train_y != train_y)
    actual_poison_rate = len(poisoned_indices) / len(poisoned_train_x)

    trigger_value = args.trigger_value
    if trigger_value is None:
        trigger_value = 1.0 if train_x.min() >= 0 and train_x.max() <= 1.0 + 1e-6 else 255.0

    attack_info = {
        "technique": "badnets",
        "trigger_size": args.trigger_size,
        "trigger_value": float(trigger_value),
        "trigger_position": args.trigger_position,
    }

    poisoning_metadata = {
        "dataset_id": "celeba",
        "dataset_variant": "badnets",
        "poison_type": "backdoor",
        "strategy": "badnets",
        "target_attribute": args.target_attribute,
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
    np.save(output_dir / "filenames.npy", train_names)
    np.save(output_dir / "test_filenames.npy", test_names)

    (output_dir / "poisoning_metadata.json").write_text(json.dumps(poisoning_metadata, indent=2))

    print(f"CelebA BadNets artifacts written to {output_dir}")
    print(f"Train={len(poisoned_train_x)}, test={len(test_x)}, poisoned={len(poisoned_indices)}")
    print(f"Poison rate: requested={args.poison_rate:.4f}, actual={actual_poison_rate:.4f}")
    print(
        f"Target={args.target_class}, trigger={args.trigger_size}x{args.trigger_size}, "
        f"position={args.trigger_position}, value={trigger_value}"
    )


if __name__ == "__main__":
    main()