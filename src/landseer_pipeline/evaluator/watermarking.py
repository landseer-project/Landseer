import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
from tqdm import tqdm
import argparse
from config_model import config

def find_uchida_style_conv2d(model):
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    
    if not conv_layers:
        return None

    # Prefer second-last conv if exists, else fallback to last
    if len(conv_layers) >= 2:
        selected_idx = -2
    else:
        selected_idx = -1

    selected_name, selected_layer = conv_layers[selected_idx]
    # print(f"[+] Selected Conv2D layer for embedding: {selected_name}")
    return selected_layer

def decode_watermark(conv_layer, wm_matrix, wm_bits):
    with torch.no_grad():
        weight = conv_layer.weight
        mean_weight = weight.mean(dim=[2, 3])
        flat = mean_weight.view(1, -1)
        projection = torch.sigmoid(flat @ wm_matrix)
        decoded_bits = (projection > 0.5).float()
        accuracy = (decoded_bits == wm_bits).float().mean().item()
        return accuracy, decoded_bits.cpu().numpy()

def finetune(model, device, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"\n[Finetuning] Epoch {epoch+1}/{epochs}")
        total, correct = 0, 0
        pbar = tqdm(train_loader)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            pbar.set_description(f"Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="result/model_with_watermark.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wm_matrix", type=str, default="result/wm_matrix.npy")
    parser.add_argument("--wm_bits", type=str, default="result/wm_bits.npy")
    parser.add_argument("--output_model", type=str, default="result/model_finetuned.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # === Load model and watermark
    model = config().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # conv_layer = find_first_conv2d(model)
    conv_layer = find_uchida_style_conv2d(model)
    if conv_layer is None:
        print("[!] No Conv2d layer found.")
        return

    wm_matrix = torch.tensor(np.load(args.wm_matrix), dtype=torch.float32).to(device)
    wm_bits = torch.tensor(np.load(args.wm_bits), dtype=torch.float32).to(device)

    acc_before, _ = decode_watermark(conv_layer, wm_matrix, wm_bits)
    print(f"\n[+] Watermark decoding BEFORE fine-tuning: {acc_before * 100:.2f}%")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    finetune(model, device, train_loader, optimizer, args.epochs)

    acc_after, decoded = decode_watermark(conv_layer, wm_matrix, wm_bits)
    print(f"\n[+] Watermark decoding AFTER fine-tuning: {acc_after * 100:.2f}%")