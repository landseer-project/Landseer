# file: post_training_defense/fine_pruning.py
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config_model import config

def neuron_activation_scores(model, loader, device):
    """
    Example function that returns average activation magnitude per neuron
    in the final convolutional layer (or the entire network).
    For simplicity, let's assume we measure activation in the penultimate layer.
    """
    # We'll store activation sums for each neuron (feature dimension).
    activation_sum = None
    samples_count = 0

    # Hook to capture penultimate layer's output
    # Modify as needed to match your architecture.
    def hook_fn(module, input_, output):
        nonlocal activation_sum
        # output shape: [batch, features, ...] – e.g. [batch, 256] if flattened
        out_data = output.detach()
        # Flatten all spatial dims if they're present, except the feature dimension
        if len(out_data.shape) > 2:
            out_data = out_data.mean(dim=[2, 3])  # average spatially

        if activation_sum is None:
            activation_sum = torch.zeros(out_data.shape[1], device=device)
        activation_sum += out_data.abs().sum(dim=0)

    # Register a forward hook on your chosen layer
    # Example: let's pick the second-to-last module in the model
    # For a more robust approach, locate your penultimate conv or linear layer explicitly.
    modules = list(model.modules())
    # E.g., let's pick the layer before the final linear
    # Adjust index as needed: modules[-2] if the last is nn.Linear, etc.
    chosen_layer = modules[-2]
    handle = chosen_layer.register_forward_hook(hook_fn)

    # Run the model on the entire loader
    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            model(X)
            samples_count += X.size(0)

    # Remove hook
    handle.remove()

    # Convert activation sums to average activation per neuron
    activation_sum /= float(samples_count)
    return activation_sum

def prune_neurons(model, keep_indices):
    """
    Zeroes out or removes neurons not in keep_indices.
    For simplicity, we show a 'soft prune' approach by setting weights to 0.
    A more advanced approach might remove the neurons entirely and reshape layers.
    """
    # Example: prune the final linear layer’s in_features or the penultimate layer.
    # Adjust to your architecture.
    modules = list(model.modules())
    layer_to_prune = modules[-1]  # Suppose this is the penultimate layer we measured

    if isinstance(layer_to_prune, nn.Linear):
        with torch.no_grad():
            # Weight shape: [out_features, in_features]
            W = layer_to_prune.weight
            B = layer_to_prune.bias
            # We keep only certain input neurons
            mask = torch.zeros_like(W)
            mask[:, keep_indices] = 1.0
            W *= mask

            # (Optionally) zero out bias elements if you want to align with pruned neurons
            # e.g., if in_features == out_features, but typically you'd keep all biases
            # that correspond to out_features. It's simpler not to prune biases in many cases.

    elif isinstance(layer_to_prune, nn.Conv2d):
        # If pruning filters in a conv layer, you'd do something else:
        # keep_indices would correspond to channels
        with torch.no_grad():
            mask = torch.zeros_like(layer_to_prune.weight)
            # shape: [out_channels, in_channels, kH, kW]
            mask[keep_indices, :, :, :] = 1.0
            layer_to_prune.weight *= mask
            # Potentially prune biases too: layer_to_prune.bias[keep_indices] = 0
    else:
        print("Warning: chosen layer not a Linear or Conv2d. Adjust code as needed.")

def main():
    parser = argparse.ArgumentParser(description="Fine-Pruning for Backdoor Removal (Post-training)")
    parser.add_argument("--prune-percentage", type=float, default=0.2,
                        help="Fraction of neurons to prune (0.2 = 20%)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load your trained model
    # This example is a placeholder. Replace with your actual architecture or
    # load the same definition used in the training script.
    model = config()
    model.load_state_dict(torch.load("/output/model.pt", map_location=device))
    model.to(device)

    # 2. Load clean data for measuring neuron activation
    data_np = np.load("/output/data.npy")
    labels_np = np.load("/output/labels.npy")
    data_tensor = torch.from_numpy(data_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()

    dataset = TensorDataset(data_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Compute average activation scores per neuron
    activation_scores = neuron_activation_scores(model, loader, device)
    # activation_scores: shape [num_neurons]
    # We want to prune the neurons with the smallest activation (lowest usage).

    # 5. Determine which neurons to keep
    num_neurons = activation_scores.shape[0]
    k = int(num_neurons * (1 - args.prune_percentage))
    # Indices of neurons sorted by activation ascending => first part is least activated
    sorted_indices = torch.argsort(activation_scores)
    keep_indices = sorted_indices[k:]  # keep the top (num_neurons-k) active neurons

    # 6. Prune the model
    prune_neurons(model, keep_indices)

    # 8. Save the pruned model
    torch.save(model.state_dict(), "/output/model.pt")

if __name__ == "__main__":
    main()

