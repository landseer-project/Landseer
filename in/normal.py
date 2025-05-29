import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from config_model import config


def ensure_dir_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        losses.append(loss.item())
    
    accuracy = 100 * correct / total
    avg_loss = np.mean(losses)
    
    # print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset from numpy files
    data_np = np.load("/output/data.npy")
    labels_np = np.load("/output/labels.npy")
    test_data_np = np.load("/output/test_data.npy")
    test_labels_np = np.load("/output/test_labels.npy")

    # Convert to Torch tensors
    data_tensor = torch.from_numpy(data_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()
    test_data_tensor = torch.from_numpy(test_data_np).float()
    test_labels_tensor = torch.from_numpy(test_labels_np).long()

    # Create DataLoaders
    train_dataset = TensorDataset(data_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
   
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = config().to(device)  # Assuming CNN is defined in config_model.py
    #model = config().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, epoch, device
        )

    print(f"Training complete for {args.epochs} epochs.")

    # Evaluation
    test_acc = test(model, test_loader, device)
    print("Test Complete.")
    
    # Save final model
    os.makedirs("/output", exist_ok=True)
    torch.save(model.state_dict(), "/output/model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training with numpy data")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum (default: 0.9)")
    
    args = parser.parse_args()
    
    main(args)