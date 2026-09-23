"""Downloads MNIST and saves it as the .npy files Landseer expects.

There was no MNIST container before this. configs/datasets.yaml already had an
mnist entry but it had an empty default_image and a TODO saying the container
still needed to be made, so any pipeline using mnist had nothing to run.

This copies what containers/datasets/cifar10 does so the output format matches.
Everything here is CPU only, it's just a download and a reshape.
"""

import argparse
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as transforms


# these four names come from required_files in configs/datasets.yaml
REQUIRED = ("data.npy", "labels.npy", "test_data.npy", "test_labels.npy")


def _already_prepared(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST artifacts for Landseer")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--download-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # skip the download if someone already ran this
    if _already_prepared(output_dir):
        print(f"MNIST artifacts already exist in {output_dir}")
        return

    download_dir = Path(args.download_dir) if args.download_dir else output_dir / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # ToTensor gives us CHW float32 scaled to [0, 1], same as the cifar10
    # container does. MNIST is grayscale so the shape ends up (1, 28, 28)
    # instead of cifar10's (3, 32, 32).
    to_tensor = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST(
        root=str(download_dir), train=True, download=True, transform=to_tensor
    )
    test_ds = torchvision.datasets.MNIST(
        root=str(download_dir), train=False, download=True, transform=to_tensor
    )

    train_x = np.stack([np.asarray(img) for img, _ in train_ds]).astype(np.float32)
    train_y = np.asarray([label for _, label in train_ds], dtype=np.int64)
    test_x = np.stack([np.asarray(img) for img, _ in test_ds]).astype(np.float32)
    test_y = np.asarray([label for _, label in test_ds], dtype=np.int64)

    np.save(output_dir / "data.npy", train_x)
    np.save(output_dir / "labels.npy", train_y)
    np.save(output_dir / "test_data.npy", test_x)
    np.save(output_dir / "test_labels.npy", test_y)
    # printing the shapes so it's obvious from the logs if something looks wrong
    print(
        f"MNIST artifacts written to {output_dir}: "
        f"train={train_x.shape}, test={test_x.shape}, dtype={train_x.dtype}, "
        f"range=[{train_x.min():.3f}, {train_x.max():.3f}]"
    )


if __name__ == "__main__":
    main()
