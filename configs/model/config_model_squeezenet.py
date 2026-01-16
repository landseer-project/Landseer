"""squeezenet in pytorch
[1] Song Han, Jeff Pool, John Tran, William J. Dally
    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            #nn.BatchNorm2d(squzee_channel),
            nn.GroupNorm(16, squzee_channel),
            nn.ReLU(inplace=False)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            #nn.BatchNorm2d(int(out_channel / 2)),
            nn.GroupNorm(16, int(out_channel / 2)),
            nn.ReLU(inplace=False)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            #nn.BatchNorm2d(int(out_channel / 2)),
            nn.GroupNorm(16, int(out_channel / 2)),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class SqueezeNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=10):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            #nn.BatchNorm2d(96),
            nn.GroupNorm(32, 96),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x

def config(class_num=10):
    return SqueezeNet(class_num=class_num)





#import math
#import torch
#import torch.nn as nn
#from collections import OrderedDict
#
#
#__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']
#
#
#
#class Fire(nn.Module):
#
#    def __init__(self, inplanes, squeeze_planes,
#                 expand1x1_planes, expand3x3_planes):
#        super(Fire, self).__init__()
#        self.inplanes = inplanes
#
#        self.group1 = nn.Sequential(
#            OrderedDict([
#                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
#                ('squeeze_activation', nn.ReLU(inplace=True))
#            ])
#        )
#
#        self.group2 = nn.Sequential(
#            OrderedDict([
#                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
#                ('expand1x1_activation', nn.ReLU(inplace=True))
#            ])
#        )
#
#        self.group3 = nn.Sequential(
#            OrderedDict([
#                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
#                ('expand3x3_activation', nn.ReLU(inplace=True))
#            ])
#        )
#
#    def forward(self, x):
#        x = self.group1(x)
#        return torch.cat([self.group2(x),self.group3(x)], 1)
#
#
#class SqueezeNet(nn.Module):
#
#    def __init__(self, version=1.0, num_classes=200):
#        super(SqueezeNet, self).__init__()
#        if version not in [1.0, 1.1]:
#            raise ValueError("Unsupported SqueezeNet version {version}:"
#                             "1.0 or 1.1 expected".format(version=version))
#        self.num_classes = num_classes
#        if version == 1.0:
#            self.features = nn.Sequential(
#                nn.Conv2d(3, 96, kernel_size=7, stride=2),
#                nn.ReLU(inplace=True),
#                #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                Fire(96, 16, 64, 64),
#                Fire(128, 16, 64, 64),
#                Fire(128, 32, 128, 128),
#                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
#                Fire(256, 32, 128, 128),
#                Fire(256, 48, 192, 192),
#                Fire(384, 48, 192, 192),
#                Fire(384, 64, 256, 256),
#                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
#                Fire(512, 64, 256, 256),
#            )
#        else:
#            self.features = nn.Sequential(
#                nn.Conv2d(3, 64, kernel_size=3, stride=2),
#                nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                Fire(64, 16, 64, 64),
#                Fire(128, 16, 64, 64),
#                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                Fire(128, 32, 128, 128),
#                Fire(256, 32, 128, 128),
#                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                Fire(256, 48, 192, 192),
#                Fire(384, 48, 192, 192),
#                Fire(384, 64, 256, 256),
#                Fire(512, 64, 256, 256),
#            )
#        # Final convolution is initialized differently form the rest
#        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
#        self.classifier = nn.Sequential(
#            nn.Dropout(p=0.5),
#            final_conv,
#            nn.ReLU(inplace=True),
#            nn.AvgPool2d(13)
#        )
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                gain = 2.0
#                if m is final_conv:
#                    m.weight.data.normal_(0, 0.01)
#                else:
#                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
#                    u = math.sqrt(3.0 * gain / fan_in)
#                    m.weight.data.uniform_(-u, u)
#                if m.bias is not None:
#                    m.bias.data.zero_()
#
#    def forward(self, x):
#        x = self.features(x)
#        x = self.classifier(x)
#        return x.view(x.size(0), self.num_classes)
#
#def squeezenet1_0(num_classes=200):
#    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
#    accuracy with 50x fewer parameters and <0.5MB model size"
#    <https://arxiv.org/abs/1602.07360>`_ paper.
#    """
#    model = SqueezeNet(version=1.0, num_classes=num_classes)
#    return model
#
#
#def squeezenet1_1(num_classes=200):
#    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
#    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
#    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
#    than SqueezeNet 1.0, without sacrificing accuracy.
#    """
#    model = SqueezeNet(version=1.1, num_classes=num_classes)
#    return model