import torch.nn as nn


class WideDPCIFARModel(nn.Module):
    """
    Wider CNN model adapted for DP training.
    Includes:
    - GroupNorm instead of BatchNorm
    - SiLU activations
    - Dropout
    - Three convolutional layers
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=1),  # 32 -> 28
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=False),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=False),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 14 -> 14
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=False),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.AdaptiveAvgPool2d((3, 3))  # 7 -> 3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128 * 3 * 3 = 1152
            nn.Linear(1152, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.SiLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def config():
    return WideDPCIFARModel()



# config_model.py. in dp, this gave 49% accuracy
# import torch.nn as nn

# class DPCIFARModel(nn.Module):
#     """
#     Wider CNN model for CIFAR-10 with DP training compatibility.
#     Two convolutional layers (128 filters), followed by adaptive pooling and MLP.
#     """
#     def __init__(self, num_classes: int = 10):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 128, kernel_size=5, stride=1),  # 32 → 28
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(2),                             # 28 → 14

#             nn.Conv2d(128, 128, kernel_size=5, stride=1),  # 14 → 10
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(2),                               # 10 → 5

#             nn.AdaptiveAvgPool2d((3, 3)),  # 5x5 → 3x3
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),                  # 128 × 3 × 3 = 1152
#             nn.Linear(1152, 512),
#             nn.ReLU(inplace=False),
#             nn.Linear(512, 512),
#             nn.ReLU(inplace=False),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# def config():
#     """Factory function for DPCIFARModel."""
#     return DPCIFARModel()



# #For In DP
# # config_model.py
# import torch.nn as nn


# class DPCIFARModel(nn.Module):
#     """
#     Abadi et al. CNN with an extra AdaptiveAvgPool so it works for any
#     input ≥ 32×32.  Output after conv-stack is forced to 3×3, so the
#     dense part stays 64*3*3 = 576 no matter what.
#     """
#     def __init__(self, num_classes: int = 10):
#         super().__init__()

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, stride=1),  # 32→28
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(2),                            # 28→14

#             nn.Conv2d(64, 64, kernel_size=5, stride=1), # 14→10
#             nn.ReLU(inplace=False),
#             nn.MaxPool2d(2),                            # 10→5

#             # NEW → force 5×5 → 3×3 regardless of input size
#             nn.AdaptiveAvgPool2d((3, 3)),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),                  # 64 × 3 × 3 = 576
#             nn.Linear(576, 384),
#             nn.ReLU(inplace=False),
#             nn.Linear(384, 384),
#             nn.ReLU(inplace=False),
#             nn.Linear(384, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# def config():
#     """Factory to keep the same API you already use elsewhere."""
#     return DPCIFARModel()


#For post and all
# from torchvision.models import resnet18
# import torch.nn as nn

# def config():
#     """Return a ResNet-18 model adapted for CIFAR-10 (32x32)."""
#     model = resnet18(weights=None)  # Set to None to train from scratch
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     model.maxpool = nn.Identity()  # Remove unnecessary maxpool for small images
#     model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for CIFAR-10
#     return model



# import torch
# import torch.nn as nn
# from torchvision.models import resnet18


# def config():
#     """Return a ResNet-18 model configured for CIFAR-10."""
#     model = resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 output classes
    
#     return model

# # --- Step 3: Define CNN Model ---
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.fc1 = nn.Linear(32 * 8 * 8, 256)
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = x.view(-1, 32 * 8 * 8)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

