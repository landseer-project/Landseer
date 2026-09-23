"""LeNet-5 for MNIST.

We needed this because all the other configs in this folder start with
nn.Conv2d(3, ...), which means they only take 3-channel 32x32 images like
cifar10. MNIST images are 1-channel 28x28 so none of them would load.

The evaluators import this file and call config() to rebuild the model
before loading the saved weights (see containers/evals/common/model_loader.py).
"""

import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 the way it's usually written, for 1x28x28 input and 10 classes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # padding=2 makes the 28x28 input act like the 32x32 the original paper used
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28 -> 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # 28 -> 14
            nn.Conv2d(6, 16, kernel_size=5),            # 14 -> 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                            # 10 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                               # 16 * 5 * 5 = 400
            nn.Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def lenet5():
    return LeNet5()


# the evaluators look for a function called exactly config()
def config():
    return lenet5()
