"""DCRNet v21-B: v21-A with bilinear projection in the dilate path.

v21-A diagnosed cr=16 lag as ``Linear(2048, code_dim)`` projection bottleneck:
at high cr the dense projection forces too much spatial information through a
narrow Linear bottleneck. v21-B replaces it with ``ComplexLowRankEncoder``
(structured rank-1 outer-product compression), the same primitive that the
bilinear main encoder uses. Lots fewer params at high cr (24K vs 262K at
cr=16) while preserving spatial structure.

Other components are unchanged from v21-A: dilated-front FCI cascade,
SpatialGate, hsigmoid output, FiLM-from-z refine.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9, ComplexLowRankEncoder
from .dcrnet_v21a import (
    DCRDecoderRefineZ, IterativeRefineZ,
)


__all__ = ["DCRNetV21B", "DilatedFrontFCIBilinearEncoderPath", "dcrnet_v21b"]


class DilatedFrontFCIBilinearEncoderPath(nn.Module):
    """Same as v21a's encoder path but ``Linear(2048, code_dim)`` is
    replaced by a ``ComplexLowRankEncoder`` -- the (2, H, W) feature map
    after the FCI cascade is interpreted as (Re, Im) channels and
    bilinearly projected via u^H · feat · v with r_enc=64 ranks.
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 r_enc_dilate: int = 64):
        super().__init__()
        # Dilated multi-scale front
        self.d_conv1 = ConvBN(in_channels, 2, [3, 1], dilation=3)
        self.d_act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.d_conv2 = ConvBN(2, 2, [1, 3], dilation=3)
        self.d_act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        # FCI cascade
        self.conv1 = ConvBN(2, 2, 3)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = ConvBN(2, 2, [1, 9])
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3 = ConvBN(2, 2, [9, 1])

        # SpatialGate, identity-init
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.sa_conv.weight)
        nn.init.constant_(self.sa_conv.bias, 3.0)

        # Bilinear projection instead of Linear(C*H*W, code_dim)
        # Treats the (2, H, W) feature map as (Re, Im) for ComplexLowRankEncoder.
        self.proj_bilinear = ComplexLowRankEncoder(
            h, w, code_dim, r_enc=r_enc_dilate, nonlinearity=True
        )
        # LayerNorm + gate to match v20/v21a's stable mixing into z_a
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.d_conv1(x); h = self.d_act1(h)
        h = self.d_conv2(h); h = self.d_act2(h)
        h = self.conv1(h); h = self.act1(h)
        h = self.conv2(h); h = self.act2(h)
        h = self.conv3(h)
        m = torch.cat([h.max(dim=1, keepdim=True)[0], h.mean(dim=1, keepdim=True)], dim=1)
        h = h * torch.sigmoid(self.sa_conv(m))
        z = self.proj_bilinear(h)
        z = self.norm(z)
        return self.gate * z


class DCRNetV21B(DCRNetV9):
    """v21a with Linear -> ComplexLowRankEncoder for the dilate path projection."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 refine_width: int = 2, refine_K: int = 1,
                 r_enc_dilate: int = 64,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         refine_width=refine_width, refine_K=refine_K,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction

        if self.use_dilate_path:
            self.enc_dilate = DilatedFrontFCIBilinearEncoderPath(
                in_channels, h, w, code_dim, r_enc_dilate=r_enc_dilate,
            )
        self.refine = IterativeRefineZ(width=refine_width, K=refine_K, code_dim=code_dim)

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse, z)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v21b(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    d_emb: int = 64,
    d_ctx: int = 32,
    refine_width: int = 2,
    refine_K: int = 1,
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
    r_enc_dilate: int = 64,
    **kwargs,
) -> DCRNetV21B:
    return DCRNetV21B(
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
        r_enc_dilate=r_enc_dilate,
        **kwargs,
    )
