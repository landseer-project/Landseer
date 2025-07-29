from torchvision.models import resnet18
import torch.nn as nn

def config():
    """Return a ResNet-18 model adapted for CIFAR-10 (32x32)."""
    model = resnet18(weights=None)  # Set to None to train from scratch
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove unnecessary maxpool for small images
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for CIFAR-10
    return model

