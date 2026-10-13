"""Compare CIFAR-10 test accuracy of the PyTorch checkpoint and the TensorFlow SavedModel."""

import os

import numpy as np
import tensorflow as tf
import torch

from config_model_export import config


def normalize(images):
    # Same per-channel stats as ghcr.io/landseer-project/in_noop:v7.
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
    return (images - mean) / std


def pytorch_accuracy(model, images, labels, batch_size=256):
    device = torch.device("cpu")
    model = model.to(device).eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(device)
            target = labels[start : start + batch_size]
            pred = model(batch).argmax(dim=1).cpu().numpy()
            correct += int((pred == target).sum())
    return correct / len(labels)


def tensorflow_accuracy(saved_model_dir, images, labels, batch_size=256):
    model = tf.saved_model.load(saved_model_dir)
    infer = model.signatures["serving_default"]
    input_name = list(infer.structured_input_signature[1].keys())[0]
    output_name = list(infer.structured_outputs.keys())[0]
    print(f"TensorFlow input={input_name} output={output_name}")
    correct = 0
    for start in range(0, len(labels), batch_size):
        batch = images[start : start + batch_size].astype(np.float32)
        result = infer(**{input_name: tf.constant(batch)})
        logits = result[output_name].numpy()
        pred = logits.argmax(axis=1)
        correct += int((pred == labels[start : start + batch_size]).sum())
    return correct / len(labels)


def main():
    data_dir = os.environ.get("DATA_DIR", "/data")
    saved_model_dir = os.environ.get("SAVED_MODEL", "/output/model")
    weights = os.environ.get("WEIGHTS", "/input/model.pt")

    images = normalize(np.load(os.path.join(data_dir, "test_data.npy"))).astype(np.float32)
    labels = np.load(os.path.join(data_dir, "test_labels.npy")).astype(np.int64)
    print(f"Test set: {images.shape} range {images.min():.3f} {images.max():.3f}")

    model = config()
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"PyTorch load strict: missing={list(missing)} unexpected={list(unexpected)}")

    pt_acc = pytorch_accuracy(model, images, labels)
    tf_acc = tensorflow_accuracy(saved_model_dir, images, labels)
    print(f"PyTorch test accuracy: {pt_acc:.4f}")
    print(f"TensorFlow test accuracy: {tf_acc:.4f}")
    print(f"Absolute difference: {abs(pt_acc - tf_acc):.4f}")


if __name__ == "__main__":
    main()
