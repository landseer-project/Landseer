"""Export architecture for the in-noop ResNet-20 checkpoint.

Same weights as config_model.py. The training forward uses
F.avg_pool2d(out, out.size()[3]), and torch.onnx.export rejects a tensor
kernel size. Adaptive average pooling to 1x1 is the same operation for this
network (32x32 input, two stride-2 stages, 8x8 map).
"""

import torch.nn.functional as F

from config_model_resnet20 import BasicBlock, ResNet


class ExportResNet(ResNet):
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def config():
    return ExportResNet(BasicBlock, [3, 3, 3], num_classes=10)
