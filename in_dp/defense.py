import argparse
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm
from PIL import Image
from config_model import config


def ensure_dir_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

class CelebADataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, target_attr="Smiling"):
        self.img_dir = ensure_dir_exists(img_dir)
        self.transform = transform
        self.df = pd.read_csv(csv_path)
        self.target_attr = target_attr
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        
        attr = self.df.iloc[idx][self.target_attr]
        target = torch.tensor(1 if attr == 1 else 0, dtype=torch.long) 
        # target = 1 if attr == 1 else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, target
    
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
    #     print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, ε: {epsilon:.2f}")
    # else:
    #     print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
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
        # print(f"Output will be saved to: {output_dir}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    data_np = np.load("/output/data.npy")
    labels_np = np.load("/output/labels.npy")

    test_data_np = np.load("/output/test_data.npy")
    test_labels_np = np.load("/output/test_labels.npy")

    # Convert NumPy arrays -> Torch tensors
    data_tensor = torch.from_numpy(data_np).float()
    labels_tensor = torch.from_numpy(labels_np).long()

    test_data_tensor = torch.from_numpy(test_data_np).float()
    test_labels_tensor = torch.from_numpy(test_labels_np).long()

    
    # Data loaders
    # Create a Dataset + DataLoader
    train_dataset = TensorDataset(data_tensor, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
    test_loader = DataLoader(
         test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
     )
    
    # Model setup
    # model = models.resnet18(pretrained=True)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 2)  # Binary classification
    model = config().to(device)
    
    # Opacus compatibility check
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
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

    
    #Save final model
    os.makedirs("/output", exist_ok=True)

    # Save the model properly accounting for DP wrapper
    if args.enable_dp:
        # Save the actual model, not the DP wrapper
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
    parser = argparse.ArgumentParser(description="DP-SGD Training on CelebA")
    
    # Data arguments
    parser.add_argument("--img_dir", type=str,
                       help="Path to directory containing CelebA images")
    parser.add_argument("--csv_path", type=str,
                       help="Path to CelebA attributes CSV file")
    parser.add_argument("--output_dir", type=str, default="/output",
                       help="Directory to save outputs (created if doesn't exist)")
    parser.add_argument("--target_attr", type=str, default="Smiling",
                       help="Target attribute for classification")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                       help="Ratio of test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs (default: 10)")
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