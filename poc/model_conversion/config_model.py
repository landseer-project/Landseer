"""CIFAR-10 entrypoint for the in-noop proof of concept.

The repo file config_model_resnet20.py is mounted unchanged. Its ResNet
defaults to 2 classes, which cannot train on CIFAR-10 labels 0-9. This
wrapper only selects 10 classes.
"""

from config_model_resnet20 import BasicBlock, ResNet


def config():
    return ResNet(BasicBlock, [3, 3, 3], num_classes=10)
