"""
# 
# This file contains our implementation of the DRN-A model.
#  
"""

import torch
import torch.nn as nn
import numpy as np

# Dilated Convolution (arch A)
class DRN_A(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], layers=[3, 4, 6, 3]
    ):
        super(DRN_A, self).__init__()
        # layers: 3,4,6,3
        # featuers: 64, 128, 256, 512
        # self.inplanes = 64
        # Bottleneck-block: expansion = 4
        block = Bottleneck
        self.inplanes = 64

        # Level 1 
        # Skipped

        # Level 2
        self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Level 3
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layer1 (bottleneck, 64, 3, stride=1, dilation=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Layer2 (bottleneck, 128, 4, stride=2, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # Layer3 (bottleneck, 256, 6, stride=1, dilation=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        
        # Layer4 (bottleneck, 512, 3, stride=1, dilation=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.seglayer = nn.Conv2d(2048, 1, kernel_size=1, bias=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.seglayer(x)
        x = self.up(x)
        x = self.sigmoid(x)

        return x


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        # Taken from https://github.com/fyu/drn/blob/d75db2ee7070426db7a9264ee61cf489f8cf178c/drn.py#L297
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    # Taken from https://github.com/fyu/drn/blob/d75db2ee7070426db7a9264ee61cf489f8cf178c/drn.py
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def test():
    x = torch.randn((1, 3, 384, 384))
    out = torch.randn((1, 1, 384, 384))
    model = DRN_A(in_channels=3, out_channels=1)
    preds = model(x)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
    assert preds.shape == out.shape


if __name__ == "__main__":
    test()
