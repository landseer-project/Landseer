import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from config_model import config
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Custom Dataset to apply transforms on test data from .npy
class NpyCIFAR10TestDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].numpy().astype(np.uint8)  # [C, H, W] or [H, W, C]
        img = np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else img  # Ensure [H, W, C]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--input_dir", type=str, default="/data")
    parser.add_argument("--output", type=str, default="/output")
    args = parser.parse_args()

    # Load test data from .npy
    test_data = np.load(os.path.join(args.input_dir, "test_data.npy"))
    test_labels = np.load(os.path.join(args.input_dir, "test_labels.npy"))

    # Convert to tensors and permute from NHWC to NCHW
    test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2).contiguous()  # [N, C, H, W]
    test_labels = torch.from_numpy(test_labels).long()

    # Define normalization transform (same as training)
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # Ensure image becomes float32 tensor in [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Apply transform through custom dataset class
    test_dataset = NpyCIFAR10TestDataset(test_data, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load trained model
    model = config()
    model.load_state_dict(torch.load(os.path.join(args.input_dir, "model.pt"), map_location="cpu"))
    model.eval()

    # Evaluate without DP noise
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print("Non-DP Accuracy: {:.4f}".format(correct / total))

    # Add DP noise during inference
    def add_dp_noise(model, dataloader, epsilon, delta):
        with torch.no_grad():
            all_probs = []
            all_labels = []
            for images, labels in dataloader:
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
                all_labels.append(labels)
            all_probs = torch.cat(all_probs)
            all_labels = torch.cat(all_labels)

        sensitivity = 1.0
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = torch.normal(0, sigma, size=all_probs.shape)
        noisy_probs = all_probs + noise
        noisy_preds = noisy_probs.argmax(dim=1)

        return noisy_preds.numpy(), all_labels.numpy(), epsilon, delta

    # Run DP evaluation
    dp_preds, true_labels, epsilon, delta = add_dp_noise(
        model, test_loader, args.epsilon, args.delta
    )

    accuracy = accuracy_score(true_labels, dp_preds)
    print(f"DP Results (ε={epsilon:.2f}, δ={delta:.1e}):")
    print(f"- Accuracy: {accuracy:.4f}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'privacy_metrics.txt'), 'w') as f:
        f.write(f"epsilon={args.epsilon}\n")
        f.write(f"delta={args.delta}\n")
        f.write(f"dp_accuracy={accuracy:.4f}\n")

if __name__ == "__main__":
    main()



# import argparse
# import os
# import torch
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import accuracy_score
# from config_model import config
# import torch.nn.functional as F

# def main():
#     # Arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epsilon", type=float, default=3.0)
#     parser.add_argument("--delta", type=float, default=1e-5)
#     parser.add_argument("--input_dir", type=str, default="/data")
#     parser.add_argument("--output", type=str, default="/output")
#     args = parser.parse_args()

#     # Load CIFAR-10 test data
#     test_data = np.load(os.path.join(args.input_dir, "test_data.npy"))
#     test_labels = np.load(os.path.join(args.input_dir, "test_labels.npy"))
    
#     # Convert to tensors and normalize
#     test_data = torch.from_numpy(test_data).float()

#     #Uncomment and normalize if the saved data is not normalized
#     test_data = (test_data / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

#     if test_data.shape[1] != 3:
#         test_data = test_data.permute(0, 3, 1, 2)
#     test_labels = torch.from_numpy(test_labels).long()
#     test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64)


#     # Load model
#     model = config()
#     model.load_state_dict(torch.load(os.path.join(args.input_dir, "model.pt"), map_location="cpu"))
#     model.eval()

#     # Evaluate without DP noise
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             outputs = model(images)
#             predicted = outputs.argmax(dim=1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#         print("Non-DP Accuracy: {:.4f}".format(correct / total))


#     def add_dp_noise(model, dataloader, epsilon, delta):
#         with torch.no_grad():
#             all_probs = []
#             all_labels = []
#             for images, labels in dataloader:
#                 logits = model(images)
#                 probs = F.softmax(logits, dim=1)
#                 all_probs.append(probs)
#                 all_labels.append(labels)
            
#             all_probs = torch.cat(all_probs)
#             all_labels = torch.cat(all_labels)

#         # Estimate L1 sensitivity (between outputs of two different samples)
#         sensitivity = 1.0  # Reasonable assumption for probability outputs (L1 norm <= 2)
    

#         # Add gaussian noise
#         scale = sensitivity / epsilon
#         noise = torch.normal(0, scale, size=all_probs.shape)
#         noisy_probs = all_probs + noise
#         noisy_preds = noisy_probs.argmax(dim=1)

#         # Manually return the given epsilon and delta (since no DP training was done)
#         return noisy_preds.numpy(), all_labels.numpy(), epsilon, delta

#     # Run DP evaluation
#     dp_preds, true_labels, epsilon, delta = add_dp_noise(
#         model, test_loader, args.epsilon, args.delta
#     )

#     accuracy = accuracy_score(true_labels, dp_preds)
#     print(f"DP Results (ε={epsilon:.2f}, δ={delta:.1e}):")
#     print(f"- Accuracy: {accuracy:.4f}")

#     # Save results
#     print(f"Saving to: {os.path.abspath(os.path.join(args.output, 'privacy_metrics.txt'))}")
#     with open(os.path.join(args.output, 'privacy_metrics.txt'), 'w') as f:
#         f.write(f"epsilon={args.epsilon}\n")
#         f.write(f"delta={args.delta}\n")
#         f.write(f"dp_accuracy={accuracy:.4f}\n")

# if __name__ == "__main__":
#     main()

