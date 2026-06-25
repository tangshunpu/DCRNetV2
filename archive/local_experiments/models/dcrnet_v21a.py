"""DCRNet v21-A: v20 + dilated-front encoder + z-conditioned refine (FiLM).

Two changes from v20:
1. enc_dilate: prepend a dilated 3x1/1x3 (d=3) pair before the FCI cascade,
   so multi-scale local structure is extracted *first*, then FCI integrates
   it over long range. (CLNet's encoder1 cascade is FCI-only and misses
   the dilation-style multi-scale extraction that helped v9u at cr=16.)

2. refine: each ``DCRDecoderRefineZ`` stage takes the codeword z in addition
   to the spatial coarse reconstruction, and FiLM-modulates its internal
   fused feature map. Lets refine add what the rank-R outer product
   decomposition can't represent (global z statistics that don't fit
   the rank-1 structure).
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN, Channel_shuffle
from .dcrnet_v9 import DCRNetV9
from .dcrnet_v20 import FCIEncoderPath


__all__ = ["DCRNetV21A", "DilatedFrontFCIEncoderPath",
           "DCRDecoderRefineZ", "IterativeRefineZ", "dcrnet_v21a"]


class DilatedFrontFCIEncoderPath(nn.Module):
    """Dilated-front FCI cascade replacement for ``DilateEncoderPath``.

    Order: dilated 3x1(d=3) + 1x3(d=3) -> FCI cascade (3x3 + 1x9 + 9x1)
    -> SpatialGate -> Linear projection. Dilated layers go first so the
    multi-scale extraction sees the raw 2-channel input, and the FCI
    cascade then integrates the multi-scale features over long range.
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int):
        super().__init__()
        # Dilated multi-scale front (d=3 for ~7-pixel effective receptive)
        self.d_conv1 = ConvBN(in_channels, 2, [3, 1], dilation=3)
        self.d_act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.d_conv2 = ConvBN(2, 2, [1, 3], dilation=3)
        self.d_act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        # FCI cascade (CLNet encoder1)
        self.conv1 = ConvBN(2, 2, 3)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = ConvBN(2, 2, [1, 9])
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3 = ConvBN(2, 2, [9, 1])

        # SpatialGate, identity-init
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.sa_conv.weight)
        nn.init.constant_(self.sa_conv.bias, 3.0)

        # Linear projection + LN + gate (same as v20)
        self.proj = nn.Linear(in_channels * h * w, code_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dilated multi-scale front
        h = self.d_conv1(x); h = self.d_act1(h)
        h = self.d_conv2(h); h = self.d_act2(h)
        # FCI cascade
        h = self.conv1(h); h = self.act1(h)
        h = self.conv2(h); h = self.act2(h)
        h = self.conv3(h)
        # SpatialGate
        m = torch.cat([h.max(dim=1, keepdim=True)[0], h.mean(dim=1, keepdim=True)], dim=1)
        h = h * torch.sigmoid(self.sa_conv(m))
        z = self.proj(h.flatten(1))
        z = self.norm(z)
        return self.gate * z


class DCRDecoderRefineZ(nn.Module):
    """v8 ``DCRDecoderRefine`` with FiLM modulation from codeword z.

    Forward signature is ``forward(x, z)`` instead of ``forward(x)``. The
    z-derived (gamma, beta) pair modulates the post-concat feature map
    before the final 1x1 fusion conv. FiLM weights are zero-init so the
    stage starts at identity (gamma=0, beta=0 -> no modulation) on top of
    the gate=1e-3 residual: identical to v9u behavior at step 0.
    """

    def __init__(self, width: int, code_dim: int):
        super().__init__()
        assert width >= 2 and width % 2 == 0
        groups = max(1, width // 2)
        self.path1 = nn.Sequential(OrderedDict([
            ("conv_3x3", ConvBN(2, width, 3, dilation=2)),
            ("prelu1", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x5", ConvBN(width, width, [3, 1], groups=groups, dilation=3)),
            ("shuffle1", Channel_shuffle(groups)),
            ("prelu2", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_5x1", ConvBN(width, width, [1, 3], groups=groups, dilation=3)),
            ("shuffle2", Channel_shuffle(groups)),
            ("prelu4", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x1", ConvBN(width, 2, 3)),
        ]))
        self.path2 = nn.Sequential(
            ConvBN(2, width, [1, 3]),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [5, 1], groups=groups),
            Channel_shuffle(groups),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [1, 5], groups=groups),
            Channel_shuffle(groups),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, 2, [3, 1]),
        )
        self.prelu = nn.PReLU(num_parameters=4, init=0.3)
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv1x1 = ConvBN(4, 2, 1)
        self.gate = nn.Parameter(torch.tensor(1e-3))

        # FiLM from z: produces (gamma, beta) for the 4 fused channels.
        # Zero-init -> identity at step 0.
        self.film_z = nn.Linear(code_dim, 2 * 4)
        nn.init.zeros_(self.film_z.weight)
        nn.init.zeros_(self.film_z.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self.prelu(torch.cat((self.path1(x), self.path2(x)), dim=1))   # (B, 4, H, W)
        gb = self.film_z(z)                                                    # (B, 8)
        gamma, beta = gb.chunk(2, dim=-1)                                      # (B, 4) each
        out = out * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        return self.prelu2(x + self.gate * self.conv1x1(out))


class IterativeRefineZ(nn.Module):
    """Stack of K ``DCRDecoderRefineZ`` stages, each receiving z."""

    def __init__(self, width: int, K: int, code_dim: int):
        super().__init__()
        assert K >= 1
        self.stages = nn.ModuleList(
            [DCRDecoderRefineZ(width=width, code_dim=code_dim) for _ in range(K)]
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x, z)
        return x


class DCRNetV21A(DCRNetV9):
    """v9u + dilated-front FCI encoder + z-conditioned refine + hsigmoid."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 refine_width: int = 2, refine_K: int = 1,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         refine_width=refine_width, refine_K=refine_K,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction

        # Swap encoder dilate path -> dilated-front FCI
        if self.use_dilate_path:
            self.enc_dilate = DilatedFrontFCIEncoderPath(in_channels, h, w, code_dim)

        # Replace refine with z-conditioned variant
        self.refine = IterativeRefineZ(width=refine_width, K=refine_K, code_dim=code_dim)

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse, z)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v21a(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    d_emb: int = 64,
    d_ctx: int = 32,
    refine_width: int = 2,         # v9u uses 2
    refine_K: int = 1,             # v9u uses 1
    use_dilate_path: bool = True,
    use_film: bool = False,
    complex_encoder: bool = True,
    hybrid_decoder: bool = True,
    hybrid_direct_ranks: int = 32,
    hybrid_extra_ranks: int = 16,
    hybrid_extra_d_emb: int = 16,
    hybrid_gate_bias: float = -2.0,
    hybrid_alpha_init: float = 1e-2,
    enc_fuse_mlp: int = 64,
    **kwargs,
) -> DCRNetV21A:
    return DCRNetV21A(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        d_emb=d_emb, d_ctx=d_ctx,
        refine_width=refine_width, refine_K=refine_K,
        use_dilate_path=use_dilate_path,
        use_film=use_film,
        complex_encoder=complex_encoder,
        hybrid_decoder=hybrid_decoder,
        hybrid_direct_ranks=hybrid_direct_ranks,
        hybrid_extra_ranks=hybrid_extra_ranks,
        hybrid_extra_d_emb=hybrid_extra_d_emb,
        hybrid_gate_bias=hybrid_gate_bias,
        hybrid_alpha_init=hybrid_alpha_init,
        enc_fuse_mlp=enc_fuse_mlp,
        h=h, w=w, total_size=total_size,
        **kwargs,
    )
