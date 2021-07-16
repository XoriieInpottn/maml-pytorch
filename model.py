#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-07-14
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True, non_linear=True):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1)
        self.pool = nn.MaxPool2d((3, 3), (2, 2), 1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if non_linear else None

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = self.pool(h)
        if self.bn:
            h = self.bn(h)
        if self.relu:
            h = self.relu(h)
        return h


class Model(nn.Module):

    def __init__(self, image_size, num_classes):
        super(Model, self).__init__()
        self.layer1 = Layer(3, 32)
        image_size = math.ceil(image_size / 2.0)
        self.layer2 = Layer(32, 32)
        image_size = math.ceil(image_size / 2.0)
        self.layer3 = Layer(32, 32)
        image_size = math.ceil(image_size / 2.0)
        self.layer4 = Layer(32, 32)
        image_size = math.ceil(image_size / 2.0)

        self._flat_size = image_size * image_size * 32
        self.fc = nn.Linear(self._flat_size, num_classes)

    def forward(self, x: torch.Tensor):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = h.reshape(-1, self._flat_size)

        h = self.fc(h)
        return h


def cross_entropy(pred, true):
    pred = F.softmax(pred, 1)
    loss = -true * torch.log(pred + 1e-10) - (1.0 - true) * torch.log(1.0 - pred + 1e-10)
    loss = loss.sum(1).mean()
    return loss
