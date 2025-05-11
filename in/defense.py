import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from trades import trades_loss  # Ensure this is your file with the TRADES implementation
import argparse
from config_model import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TRADES Robust Training for CIFAR-10 (CNN Version)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (increased from 10 to 50)")
    parser.add_argument("--epsilon", type=float, default=0.031,
                        help="Radius of adversarial perturbation for TRADES (lowered from 0.3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    # A small CNN for CIFAR-10 (instead of a single linear layer).
    # Typical structure: 2 conv layers, ReLU, and pooling,
    # then a final linear layer to output 10 classes.
    # ---------------------------------------------------------------
    model = config().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # === Load feature-squeezed CIFAR-10 data from .npy ===
    # data_np: shape [N, 3, 32, 32]
    # labels_np: shape [N]
    data_np = np.load("/output/data.npy")
    labels_np = np.load("/output/labels.npy")

    # Convert NumPy arrays -> Torch tensors
    data_tensor = torch.from_numpy(data_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()

    # Create a Dataset + DataLoader
    train_dataset = TensorDataset(data_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # === Training Loop with TRADES ===
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            # trades_loss uses the model in both eval (for adv gen) and train mode internally
            loss = trades_loss(model, X, y, optimizer, epsilon=args.epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        avg_loss = running_loss / total_samples if total_samples else 0.0

        # Evaluate on training data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        train_accuracy = correct / total if total else 0.0

        # print(f"Epoch [{epoch+1}/{args.epochs}] - "
        #       f"Loss: {avg_loss:.4f}, "
        #       f"Train Accuracy: {train_accuracy*100:.2f}%")

    # Save final model
    torch.save(model.state_dict(), "/output/model.pt")
