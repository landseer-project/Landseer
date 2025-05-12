import numpy as np
import argparse
import os


def feature_squeezing(X, bit_depth=4):
    max_val = 2 ** bit_depth - 1
    return np.round(X * max_val) / max_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Feature Squeeze CIFAR-10 using pre-saved .npy files")
    parser.add_argument("--input-dir", default="/data",
                        help="Directory with X_train.npy, y_train.npy, etc.")
    parser.add_argument("--output", default="/output",
                        help="Output directory for squeezed .npy files")
    parser.add_argument("--bit-depth", type=int, default=4,
                        help="Bit depth for squeezing")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Load training and test data from .npy files
    X_train = np.load(os.path.join(args.input_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.input_dir, "y_train.npy"))
    X_test = np.load(os.path.join(args.input_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.input_dir, "y_test.npy"))

    print(f"Loaded training data: {X_train.shape}")
    print(f"Loaded test data: {X_test.shape}")

    # Normalize if required (optional, depending on preprocessing)
    if X_train.max() > 1.0:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    # Apply feature squeezing
    X_train_squeezed = feature_squeezing(X_train, bit_depth=args.bit_depth)

    # Save squeezed arrays
    np.save(os.path.join(args.output, "data.npy"), X_train_squeezed)
    np.save(os.path.join(args.output, "labels.npy"), y_train)
    np.save(os.path.join(args.output, "test_data.npy"), X_test)
    np.save(os.path.join(args.output, "test_labels.npy"), y_test)
    print("Saved squeezed training data to", args.output)
