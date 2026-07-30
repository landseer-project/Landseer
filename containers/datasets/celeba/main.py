"""Prepare CelebA Landseer artifacts from a local downloaded dataset tree.

Expected source layout (mounted at /source_artifacts by Landseer):
  list_attr_celeba.csv
  img_align_celeba/img_align_celeba/*.jpg   (or img_align_celeba/*.jpg)

Writes to /output:
  data.npy, labels.npy, test_data.npy, test_labels.npy,
  filenames.npy, test_filenames.npy, img_align_celeba/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


REQUIRED = (
    "data.npy",
    "labels.npy",
    "test_data.npy",
    "test_labels.npy",
    "filenames.npy",
    "test_filenames.npy",
)


def _already_prepared(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED) and (
        output_dir / "img_align_celeba"
    ).is_dir()


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
        # Landseer CelebA contract: CHW float32 in [0, 255]
        arr = np.asarray(rgb, dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))


def _load_split(
    img_dir: Path,
    file_names: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    output_img_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    valid_labels: list[int] = []
    valid_names: list[str] = []
    output_img_dir.mkdir(parents=True, exist_ok=True)

    for filename, label in zip(file_names, labels):
        name = str(filename)
        src = img_dir / name
        try:
            arr = _load_image_chw(src, image_size=image_size)
        except Exception as exc:  # noqa: BLE001 - skip corrupt/missing files
            print(f"Skipping {name}: {exc}")
            continue

        images.append(arr)
        valid_labels.append(int(label))
        valid_names.append(name)

        # Persist resized RGB jpg used by tools that read by filename
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
    parser = argparse.ArgumentParser(
        description="Prepare CelebA artifacts for Landseer from a local downloaded dataset"
    )
    parser.add_argument("--output", default="/output")
    parser.add_argument(
        "--source-artifacts-dir",
        default="/source_artifacts",
        help="Host-mounted CelebA download root (CSV + img_align_celeba)",
    )
    parser.add_argument("--target-attribute", default="Smiling")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on total samples (0 = use all). Useful for smoke tests.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _already_prepared(output_dir):
        print(f"CelebA artifacts already exist in {output_dir}")
        return

    source_root = Path(args.source_artifacts_dir)
    if not source_root.exists():
        raise FileNotFoundError(
            f"Source CelebA directory not found: {source_root}. "
            "Mount the downloaded dataset via Landseer source_artifacts_dir "
            "(or LANDSEER_DATASET_SOURCE_CELEBA)."
        )

    img_dir = _resolve_image_dir(source_root)
    attr_path = _resolve_attr_csv(source_root)

    df = pd.read_csv(attr_path)
    if "image_id" not in df.columns:
        raise ValueError(f"{attr_path} must contain an 'image_id' column")
    if args.target_attribute not in df.columns:
        raise ValueError(
            f"Attribute '{args.target_attribute}' not found in {attr_path}. "
            f"Available: {', '.join(df.columns[1:])}"
        )

    filenames = df["image_id"].astype(str).values
    # CelebA attributes are -1/1; convert to 0/1
    labels = (df[args.target_attribute].values == 1).astype(np.int64)

    if args.limit and args.limit > 0:
        filenames = filenames[: args.limit]
        labels = labels[: args.limit]

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
        img_dir, train_files, train_labels, args.image_size, out_img_dir
    )
    test_x, test_y, test_names = _load_split(
        img_dir, test_files, test_labels, args.image_size, out_img_dir
    )

    np.save(output_dir / "data.npy", train_x)
    np.save(output_dir / "labels.npy", train_y)
    np.save(output_dir / "test_data.npy", test_x)
    np.save(output_dir / "test_labels.npy", test_y)
    np.save(output_dir / "filenames.npy", train_names)
    np.save(output_dir / "test_filenames.npy", test_names)

    print(
        f"CelebA artifacts written to {output_dir}: "
        f"{len(train_names)} train, {len(test_names)} test "
        f"(attribute={args.target_attribute}, image_size={args.image_size})"
    )


if __name__ == "__main__":
    main()
