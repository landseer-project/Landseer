#################################### TRADES WITH CELEBA SUPPORTED ############################################
from __future__ import print_function
import os, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# from models.wideresnet import *
# from models.resnet import *
from config_model import config
from trades import trades_loss

# -------------------- args --------------------
parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'], help='dataset name')
parser.add_argument('--target-size', type=int, default=32, help='target image size (e.g., 32 or 64)')
parser.add_argument('--input_dir', type=str, default='/data', help='Directory containing the preprocessed .npy files')
parser.add_argument('--output', type=str, default='/output', help='Directory containing output files/ models')
parser.add_argument('--batch-size', type=int, default=128, metavar='N')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N')
parser.add_argument('--epochs', type=int, default=76, metavar='N')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--epsilon', type=float, default=0.031)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--step-size', type=float, default=0.007)
parser.add_argument('--beta', type=float, default=6.0)
parser.add_argument('--seed', type=int, default=1, metavar='S')
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
args = parser.parse_args()

# -------------------- setup --------------------
os.makedirs(args.output, exist_ok=True)
use_cuda = (not args.no_cuda) and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# -------------------- helpers --------------------
def _ensure_chw_uint8(x: np.ndarray) -> np.ndarray:
    """Accept HWC uint8, HWC float[0,1], or CHW* forms and return HWC uint8 for PIL/ToTensor."""
    if x.ndim != 3:
        raise ValueError(f'expected 3D array per image, got shape {x.shape}')
    # If CHW -> HWC
    if x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    # scale floats
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0 + 0.5).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = x.astype(np.uint8)
    return x

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = _ensure_chw_uint8(self.data[idx])  # HWC uint8
        img = Image.fromarray(img)               # PIL for torchvision transforms
        if self.transform is not None:
            img = self.transform(img)            # Tensor CxHxW float32 in [0,1]
        label = int(self.labels[idx])
        return img, label

def save_transformed_split(dataset, out_x_path, out_y_path, batch_size=512):
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=g)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    np.save(out_x_path, X)
    np.save(out_y_path, Y)

# -------------------- transforms --------------------
if args.dataset == 'celeba':
    print(f"Using CelebA transforms with target size {args.target_size}x{args.target_size}")
    transform_train = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((args.target_size, args.target_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((args.target_size, args.target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
else:
    # Default to CIFAR-10 style
    print("Using CIFAR-10 transforms")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Note: Often CIFAR TRADES code doesn't normalize here because the model or loss handles it, but if your config expects it, add it.
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

# -------------------- load NPY --------------------
train_x = np.load(os.path.join(args.input_dir, 'data.npy'))
train_y = np.load(os.path.join(args.input_dir, 'labels.npy'))
test_x  = np.load(os.path.join(args.input_dir, 'test_data.npy'))
test_y  = np.load(os.path.join(args.input_dir, 'test_labels.npy'))

trainset = NpyDataset(train_x, train_y, transform=transform_train)
testset  = NpyDataset(test_x,  test_y,  transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.test_batch_size, shuffle=False, **kwargs)

# Save one transformed pass to /output as .npy
save_transformed_split(trainset,
    os.path.join(args.output, 'data.npy'),
    os.path.join(args.output, 'labels.npy'))
save_transformed_split(testset,
    os.path.join(args.output, 'test_data.npy'),
    os.path.join(args.output, 'test_labels.npy'))

# -------------------- training / eval --------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def eval_train(model, device, loader):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            pred = out.argmax(1)
            correct += (pred == target).sum().item()
    loss = loss_sum / len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f'Training: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc*100:.0f}%)')
    return loss, acc

def eval_test(model, device, loader):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            pred = out.argmax(1)
            correct += (pred == target).sum().item()
    loss = loss_sum / len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f'Test: Average loss: {loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc*100:.0f}%)')
    return loss, acc

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 75:  lr = args.lr * 0.1
    if epoch >= 90:  lr = args.lr * 0.01
    if epoch >= 100: lr = args.lr * 0.001
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def main():
    model = config().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

    torch.save(model.state_dict(), os.path.join(args.output, 'model.pt'))
    print(f'Saved final model to {os.path.join(args.output, "model.pt")}')

if __name__ == '__main__':
    main()


