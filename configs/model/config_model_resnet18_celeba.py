import torch.nn as nn
from torchvision.models import resnet18


def config():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
