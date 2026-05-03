# main.py
import argparse
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from config_model import config
# from config_model_resnet20 import config 
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
# from opacus.accountants import RDPAccountant

logging.basicConfig(level=logging.INFO)



def save_clean_model(dp_model, path, device):
    """
    Saves a clean (unwrapped) version of a DP-wrapped model.
    """
    state_dict = dp_model.state_dict()

    # Remove _module or _module._module prefixes
    for prefix in ["_module._module.", "_module."]:
        if all(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            break  # Stop after first successful match

    clean_model = config().to(device)
    clean_model.load_state_dict(state_dict)
    torch.save(clean_model.state_dict(), path)
    print(f"Clean model saved to: {path}")





class TensorDatasetWithTransform(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float()
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def train(model, loader, opt, criterion, epoch, device, pe=None, delta=1e-5):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    eps = None
    if pe:
        try:
            eps = pe.get_epsilon(delta)
        except Exception as e:
            logging.warning(f"Could not compute ε: {e}")

    lr = opt.param_groups[0]['lr']
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(
        f"Epoch {epoch}: LR={lr:.5f}, Loss={avg_loss:.4f}, "
        f"Acc={acc:.2f}%, Eps={eps if eps is not None else 'N/A'}"
    )
    return eps, acc


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.dataset == "celeba":
        print(f"Using CelebA transforms with target size {args.target_size}x{args.target_size}")

        # These assume your images are already float tensors in [0,1] (or similar)
        # If your CelebA .npy is in [0,255], you MUST divide by 255 before Normalize.
        CELEBA_MEAN = (0.5, 0.5, 0.5)
        CELEBA_STD  = (0.5, 0.5, 0.5)

        transform_train = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((args.target_size, args.target_size)),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ])

        transform_test = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((args.target_size, args.target_size)),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ])

    else:
        print("Using CIFAR-10 transforms")

        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

        transform_train = transforms.Compose([
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        transform_test = transforms.Compose([
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])



    print(f"Loading data from: {args.input_dir}")
    X_train = np.load(os.path.join(args.input_dir, "data.npy"))
    Y_train = np.load(os.path.join(args.input_dir, "labels.npy"))
    X_test = np.load(os.path.join(args.input_dir, "test_data.npy"))
    Y_test = np.load(os.path.join(args.input_dir, "test_labels.npy"))

    train_ds = TensorDatasetWithTransform(X_train, Y_train, transform_train)
    test_ds = TensorDatasetWithTransform(X_test, Y_test, transform_test)

    print(f"Train dataset size: {len(train_ds)}")


    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Test dataset size:  {len(test_ds)}")

    # === Save transformed training and test data as .npy files ===
    print("Saving transformed train/test datasets as .npy...")

    def collect_transformed_data(loader):
        X_list, Y_list = [], []
        for x, y in loader:
            X_list.append(x)
            Y_list.append(y)
        X_all = torch.cat(X_list, dim=0).cpu().numpy()
        Y_all = torch.cat(Y_list, dim=0).cpu().numpy()
        return X_all, Y_all

    X_train_tf, Y_train_tf = collect_transformed_data(train_ld)
    X_test_tf, Y_test_tf = collect_transformed_data(test_ld)

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "data1.npy"), X_train_tf)
    np.save(os.path.join(args.output, "labels1.npy"), Y_train_tf)
    np.save(os.path.join(args.output, "test_data1.npy"), X_test_tf)
    np.save(os.path.join(args.output, "test_labels1.npy"), Y_test_tf)

    print("Transformed datasets saved to", args.output)



    model = config().to(device)
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=True)

    criterion = nn.CrossEntropyLoss()


    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

   

    sample_rate = args.batch_size / len(train_ds)
    print("Differential Privacy enabled")
    print(f"Sample rate: {sample_rate:.6f}")

    privacy_engine = PrivacyEngine()

    model, optimizer, train_ld = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_ld,
        epochs=args.epochs,                 # IMPORTANT: planned epochs
        target_epsilon=args.target_epsilon, # e.g. 50
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,   # e.g. 1.2
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")


    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        eps, train_acc = train(
            model, train_ld, optimizer, criterion, epoch, device,
            privacy_engine, args.delta
        )
        scheduler.step()

        test_acc = evaluate(model, test_ld, device)
        if test_acc > best_acc:
            best_acc = test_acc

            # Save clean model
            clean_model_path = os.path.join(args.output, "model.pt")
            save_clean_model(model, clean_model_path, device)



    final_eps = privacy_engine.get_epsilon(args.delta)
    print(f"Final ε = {final_eps:.4f}, δ = {args.delta}")
    print(f"Best test accuracy: {best_acc:.2f}%")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "privacy_metrics.txt"), "w") as f:
        f.write(f"epsilon={final_eps:.4f}\n")
        f.write(f"delta={args.delta}\n")
        f.write(f"dp_accuracy={best_acc:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    parser.add_argument("--target_size", type=int, default=32, help="Only used for celeba. E.g. 32, 64 or 128.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--target_epsilon", type=float, default=50.0)
    args = parser.parse_args()
    main(args)




