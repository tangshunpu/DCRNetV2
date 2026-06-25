"""CRNet — Channel-State-Information Compression with Multi-Resolution
Convolutional Network.

Ported from https://github.com/Kylin9511/CRNet
Reference: Lu, Z., et al., "Multi-Resolution CSI Feedback With Deep Learning
in Massive MIMO System," ICC 2020.

Differences from the upstream source:
    - dropped `from utils import logger` (uses no logger here)
    - h, w, total_size are kwargs, default 32×32 / 2048 for COST2100 protocol
"""
from collections import OrderedDict
import os

import torch
import torch.nn as nn


__all__ = ["crnet", "CRNet"]


class FrobeniusNormOutput(nn.Module):
    """Project centered output to ||·||_F = 0.5 per sample, shift to [0,1].
    Matches the csi_frob data convention: x = s/2 + 0.5 where ||s||_F = 1."""
    def forward(self, x):
        n = x.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
        return x / n * 0.5 + 0.5


class FrobeniusNormOutputZ(nn.Module):
    """Frobenius projection with codeword-conditioned shrinkage α ∈ [0.5, 1]."""
    def __init__(self, codeword_dim):
        super().__init__()
        self.alpha_head = nn.Linear(codeword_dim, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
    def forward(self, x, z):
        n = x.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
        alpha = torch.sigmoid(self.alpha_head(z)).view(-1, 1, 1, 1) * 0.5 + 0.5
        return x / n * 0.5 * alpha + 0.5


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
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
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


class CRNet(nn.Module):
    def __init__(self, reduction=4, h=32, w=32, total_size=None, in_channel=2,
                 **_unused):
        super(CRNet, self).__init__()
        if total_size is None:
            total_size = in_channel * h * w
        self.h, self.w = h, w
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)
        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ]))
        if os.environ.get('DCRNET_CENTERED', '0') == '1':
            if os.environ.get('DCRNET_TANH', '0') == '1':
                self.sigmoid = nn.Tanh()
            else:
                self.sigmoid = nn.Identity()
        elif os.environ.get('DCRNET_FROBENIUS_OUTPUT', '0') == '1':
            if os.environ.get('DCRNET_FROBENIUS_ALPHA_Z', '0') == '1':
                self.sigmoid = FrobeniusNormOutputZ(total_size // reduction)
            else:
                self.sigmoid = FrobeniusNormOutput()
        else:
            self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.size()
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        z = self.encoder_fc(out.view(n, -1))
        out = self.decoder_fc(z).view(n, c, h, w)
        out = self.decoder_feature(out)
        if isinstance(self.sigmoid, FrobeniusNormOutputZ):
            out = self.sigmoid(out, z)
        else:
            out = self.sigmoid(out)
        return out


def crnet(reduction=4, expansion=1, h=32, w=32, total_size=None, **_unused):
    del expansion       # not used by CRNet
    return CRNet(reduction=reduction, h=h, w=w, total_size=total_size)
