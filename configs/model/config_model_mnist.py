"""A plain CNN for MNIST.

This was the first model we wrote for MNIST before switching to LeNet-5.
Keeping it around so we can compare a bigger model (about 1.2M parameters)
against LeNet-5 (about 62k) on the same data. Point the MODEL_CONFIG env
var at this file if you want to use it instead.

Same idea as the LeNet config: 1 channel 28x28 input, and config() is what
the evaluators call.
"""

import torch.nn as nn


class MnistCNN(nn.Module):
    """Two conv layers then two linear layers. Nothing special."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),   # 28 -> 26
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),  # 26 -> 24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 64 * 12 * 12 = 9216
            nn.Linear(9216, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def mnist_cnn():
    return MnistCNN()


def config():
    return mnist_cnn()
