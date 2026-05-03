import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import numpy as np
from config_model import config

# --- Parse Command-Line Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="/output", help="Directory to save outputs")
parser.add_argument("--input_dir", type=str, default="/data", help="Directory to load .npy data from")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.1)
args = parser.parse_args()

# --- Create Output Directory ---
os.makedirs(args.output, exist_ok=True)


# --- Data Transforms (augmentation only for training) ---
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# --- Load data from .npy ---
train_images = np.load(os.path.join(args.input_dir, "data.npy"))        # shape: (N, C, H, W)
train_labels = np.load(os.path.join(args.input_dir, "labels.npy"))
test_images = np.load(os.path.join(args.input_dir, "test_data.npy"))
test_labels = np.load(os.path.join(args.input_dir, "test_labels.npy"))

print("Train data shape:", train_images.shape)  # should be (N, 3, 32, 32)
print("Train dtype:", train_images.dtype)       # should be float32
print("Train range:", train_images.min(), train_images.max())  # should be [0, 1]



# --- Custom Dataset to apply torchvision transforms ---
from PIL import Image
class CIFAR10NPY(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images).float()  # (N, C, H, W), already in [0,1]
        self.labels = torch.tensor(labels).long()
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]  # Tensor: (C, H, W)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


    def __len__(self):
        return len(self.images)

# --- Dataloaders ---
train_dataset = CIFAR10NPY(train_images, train_labels, transform=transform_train)
test_dataset = CIFAR10NPY(test_images, test_labels, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# --- Save Transformed Data ---
def save_transformed_dataset(loader, data_filename, labels_filename):
    X_all, Y_all = [], []
    for x, y in loader:
        X_all.append(x)
        Y_all.append(y)
    X_all = torch.cat(X_all, dim=0).cpu().numpy()
    Y_all = torch.cat(Y_all, dim=0).cpu().numpy()
    np.save(os.path.join(args.output, data_filename), X_all)
    np.save(os.path.join(args.output, labels_filename), Y_all)
    print(f"Saved {data_filename} and {labels_filename} to {args.output}")

# Save transformed datasets
save_transformed_dataset(train_loader, "data.npy", "labels.npy")
save_transformed_dataset(test_loader, "test_data.npy", "test_labels.npy")



# --- Model, Optimizer, Loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = config().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# --- Training ---
def train(model, dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss:.4f} - Train Acc: {acc:.2f}%")

# --- Evaluation ---
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# --- Run Training and Evaluation ---
train(model, train_loader, args.epochs)

# Save model
torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
print("Model saved to", os.path.join(args.output, "model.pt"))

# Evaluate on test set
accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")


