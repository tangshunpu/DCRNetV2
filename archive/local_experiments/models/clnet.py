"""CLNet — Complex-input Lightweight Network for CSI Feedback.

Ported from https://github.com/SIJIEJI/CLNet
Reference: Ji, S. & Li, M., "CLNet: Complex Input Lightweight Neural Network
Designed for Massive MIMO CSI Feedback," IEEE Wireless Commun. Lett., 2021.

Differences from upstream source:
    - dropped `from utils import logger`
    - h, w, total_size are kwargs (default 32×32 / 2048 for COST2100 protocol)
    - decoder fc replaces 1×1 Conv1d (functionally equivalent)
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["clnet", "CLNet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x3', ConvBN(7, 7, [1, 3])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x1', ConvBN(7, 7, [3, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        out = self.relu(out + identity)
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                  padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CLNet(nn.Module):
    def __init__(self, reduction=4, h=32, w=32, total_size=None, in_channel=2,
                 **_unused):
        super(CLNet, self).__init__()
        if total_size is None:
            total_size = in_channel * h * w
        self.h, self.w, self.in_channel = h, w, in_channel
        self.total_size = total_size

        # Encoder ------------------------------------------------------
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 32, 1)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(34, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.sa = SpatialGate()
        self.se = SELayer(32)
        self.enc_fc = nn.Conv1d(total_size, total_size // reduction, 1)

        # Decoder ------------------------------------------------------
        self.dec_fc = nn.ConvTranspose1d(total_size // reduction, total_size, 1)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ]))
        self.hsig = hsigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n = x.size(0)
        e1 = self.encoder1(x)
        e1 = self.sa(e1)
        e2 = self.encoder2(x)
        e2 = self.se(e2)
        out = torch.cat((e1, e2), dim=1)
        out = self.encoder_conv(out)
        out = out.view(n, -1).unsqueeze(2)           # (n, total_size, 1)
        out = self.enc_fc(out)                        # (n, total_size//cr, 1)
        out = self.dec_fc(out).view(n, self.in_channel, self.h, self.w)
        out = self.decoder_feature(out)
        out = self.hsig(out)
        return out


def clnet(reduction=4, expansion=1, h=32, w=32, total_size=None, **_unused):
    del expansion       # not used by CLNet
    return CLNet(reduction=reduction, h=h, w=w, total_size=total_size)
