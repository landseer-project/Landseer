import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from PIL import Image
from config_model import config


def ensure_dir_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
        # Ensure data is in correct shape and type
        if len(data.shape) == 4:  # If already in NCHW format
            self.data = np.transpose(data, (0, 2, 3, 1))  # Convert to NHWC
        self.data = self.data.astype(np.uint8)  # Convert to uint8 for PIL
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
def train(model, train_loader, optimizer, criterion, epoch, device, privacy_engine=None, delta=1e-5):
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
    
    epsilon = None
    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
    
    return avg_loss, accuracy, epsilon


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
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if specified
    if args.output_dir:
        output_dir = ensure_dir_exists(args.output_dir)
    
    # Transformations for CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                             std=[0.2470, 0.2435, 0.2616]),
    ])
    
    # Load dataset from .npy files
    train_data = np.load("/output/data.npy")
    train_labels = np.load("/output/labels.npy")
    test_data = np.load("/output/test_data.npy")
    test_labels = np.load("/output/test_labels.npy")
    
    # # Print data shapes for debugging
    # print(f"Train data shape: {train_data.shape}, dtype: {train_data.dtype}")
    # print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    # print(f"Test data shape: {test_data.shape}, dtype: {test_data.dtype}")
    # print(f"Test labels shape: {test_labels.shape}, dtype: {test_labels.dtype}")
    
    # Create datasets
    train_dataset = CIFAR10Dataset(train_data, train_labels, transform=transform)
    test_dataset = CIFAR10Dataset(test_data, test_labels, transform=transform)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1
    )
    
    # Model setup
    model = config().to(device)
    
    # Opacus compatibility check
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum
    )
    
    # Privacy engine
    privacy_engine = None
    if args.enable_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, epsilon = train(
            model, train_loader, optimizer, criterion, epoch, device, privacy_engine, args.delta
        )

    # Evaluation
    test_acc = test(model, test_loader, device)
    
    # Save final model
    os.makedirs("/output", exist_ok=True)

    if args.enable_dp:
        torch.save(model._module.state_dict(), "/output/model.pt")
    else:
        torch.save(model.state_dict(), "/output/model.pt")
    
    # Final privacy report
    if args.enable_dp:
        final_epsilon = privacy_engine.get_epsilon(args.delta)
        print(f"\nFinal privacy cost: ε = {final_epsilon:.2f}, δ = {args.delta}")
    
    if args.enable_dp:
        print(f"Saving to: {os.path.abspath('/output/privacy_values.txt')}")
        with open('/output/privacy_values.txt', 'w') as f:
            f.write(f"epsilon={final_epsilon}\n")
            f.write(f"delta={args.delta}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DP-SGD Training on CIFAR10")
    
    # Data arguments
    parser.add_argument("--output_dir", type=str, default="/output",
                       help="Directory to save outputs (created if doesn't exist)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs (default: 45)")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum (default: 0.9)")
    
    # Differential privacy arguments
    parser.add_argument("--enable_dp", action="store_true",
                       help="Enable differential privacy")
    parser.add_argument("--noise_multiplier", type=float, default=1.1,
                       help="Noise multiplier for DP (default: 1.1)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for DP (default: 1.0)")
    parser.add_argument("--delta", type=float, default=1e-5,
                       help="Delta for DP (default: 1e-5)")
    
    args = parser.parse_args()
    
    main(args)