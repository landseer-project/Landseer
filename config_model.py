# config_model.py

import torch.nn as nn
from torchvision.models import resnet18

# def config():
#     return nn.Sequential(
#         nn.Conv2d(3, 32, 3, 1, 1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(32, 64, 3, 1, 1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Flatten(),
#         nn.Linear(64 * 8 * 8, 2),
#     )

def config():
    """Return a ResNet-18 model configured for CelebA dataset."""
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust for 2 output classes (binary classification)
    
    return model