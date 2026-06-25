"""TCLNet — Transformer-Conv hybrid for CSI feedback (Lossy_Compressor).

Adapted from https://github.com/Zijiuyang/TCLNet (Apache-2.0).

Architecture is the same skeleton as CLNet (SA + SE on parallel encoder paths,
Conv1d FC bottleneck, hsigmoid output), but the decoder CRBlock × 2 is
replaced by TCBlock × 2.  TCBlock adds a Swin-style ConvTransBlock branch on
top of CRNet's asymmetric 1×9 / 9×1 conv paths, fusing local conv features
with windowed self-attention.

Required deps:  einops, timm, compressai.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from compressai.layers import ResidualBlock


__all__ = ["tclnet", "TCLNet"]


# -------------------------------------------------------------------------
# Building blocks (copied verbatim from Zijiuyang/TCLNet/Lossy_Compressor/Modules.py)
# -------------------------------------------------------------------------


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, attn_type):
        super().__init__()
        assert attn_type in ("W", "SW")
        assert input_dim % head_dim == 0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = attn_type
        self.embedding_layer = nn.Linear(input_dim, 3 * input_dim, bias=True)
        self.linear = nn.Linear(input_dim, output_dim)
        rel_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_params = nn.Parameter(
            torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size - 1))
        nn.init.trunc_normal_(self.relative_position_params, std=0.02)

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool,
                                 device=self.relative_position_params.device)
        if shift > 0:
            s = p - shift
            attn_mask[-1, :, :s, :, s:, :] = True
            attn_mask[-1, :, s:, :, :s, :] = True
            attn_mask[:, -1, :, :s, :, s:] = True
            attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)")
        return attn_mask

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
                       p1=self.window_size, p2=self.window_size)
        h_w = x.size(1); w_w = x.size(2)
        x = rearrange(x, "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
                       p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, "b nw np (threeh c) -> threeh b nw np c",
                            c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")
        if self.type != "W":
            attn_mask = self.generate_mask(h_w, w_w, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))
        probs = torch.softmax(sim, dim=-1)
        out = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        out = rearrange(out, "h b w p c -> b w p (h c)")
        out = self.linear(out)
        out = rearrange(out, "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
                          w1=h_w, p1=self.window_size)
        if self.type != "W":
            out = torch.roll(out, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))
        return out

    def relative_embedding(self):
        cord = torch.tensor(
            np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]),
            device=self.relative_position_params.device)
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, attn_type="W"):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, attn_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, attn_type="W"):
        super().__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.trans_block = Block(trans_dim, trans_dim, head_dim, window_size, drop_path, attn_type)
        self.conv1_1 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(conv_dim + trans_dim, conv_dim + trans_dim, 1, 1, 0, bias=True)
        self.conv_block = ResidualBlock(conv_dim, conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange("b c h w -> b h w c")(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        return x + res


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                padding=padding, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class TCBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ("conv3x3", ConvBN(2, 11, 7)),
            ("relu5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9", ConvBN(11, 11, [1, 9])),
            ("conv9x1", ConvBN(11, 11, [9, 1])),
            ("ConvTrans5", ConvTransBlock(conv_dim=5, trans_dim=6, head_dim=1,
                                            window_size=4, drop_path=0, attn_type="SW")),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ("conv3x3_3", ConvBN(2, 11, 5)),
            ("relu6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x5", ConvBN(11, 11, [1, 11])),
            ("conv5x1", ConvBN(11, 11, [11, 1])),
            ("ConvTrans6", ConvTransBlock(conv_dim=5, trans_dim=6, head_dim=1,
                                            window_size=4, drop_path=0, attn_type="SW")),
        ]))
        self.conv1x1 = ConvBN(22, 2, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        return self.relu(out + identity)


# -------------------------------------------------------------------------
# CLNet-style assembly (encoder + bottleneck + decoder × TCBlock × 2)
# -------------------------------------------------------------------------


class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=groups, bias=bias)
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
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, 7, stride=1, padding=3, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class TCLNet(nn.Module):
    def __init__(self, reduction=4, h=32, w=32, total_size=None, in_channel=2, **_unused):
        super().__init__()
        if total_size is None:
            total_size = in_channel * h * w
        self.h, self.w, self.in_channel = h, w, in_channel
        self.total_size = total_size

        # Encoder (same skeleton as CLNet)
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

        # Decoder — replace CLNet's CRBlock × 2 with TCBlock × 2
        self.dec_fc = nn.ConvTranspose1d(total_size // reduction, total_size, 1)
        self.decoder_feature = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("TCBlock1", TCBlock()),
            ("TCBlock2", TCBlock()),
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
        out = out.view(n, -1).unsqueeze(2)
        out = self.enc_fc(out)
        out = self.dec_fc(out).view(n, self.in_channel, self.h, self.w)
        out = self.decoder_feature(out)
        out = self.hsig(out)
        return out


def tclnet(reduction=4, expansion=1, h=32, w=32, total_size=None, **_unused):
    del expansion
    return TCLNet(reduction=reduction, h=h, w=w, total_size=total_size)
