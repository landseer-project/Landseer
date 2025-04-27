# config_model.py

import torch.nn as nn

# def config():
#     return nn.Sequential(
#         nn.Conv2d(3, 32, 3, 1, 1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(32, 64, 3, 1, 1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Flatten(),
#         nn.Linear(64 * 8 * 8, 10)
#     )

def config():
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10) # Assuming 10 classes for CIFAR-10
    return model