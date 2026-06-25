"""DCRNet v20: v9u + CLNet-inspired FCI cascade + SpatialGate + hsigmoid.

Single targeted change to v9u (unified): replace ``DCREncoderBlock`` in the
dilate path with CLNet's 3-conv FCI cascade (3x3 -> 1x9 -> 9x1, all 2->2),
add a SpatialGate on its output, and switch the final clamp to hsigmoid.

Rationale: every other CLNet-style add (SE on codeword, SA+SE warm-restart,
SA_enc+SE_z from-scratch) has already been tested on v9u and didn't help.
The one CLNet feature v9u has never had is literal 1xK / Kx1 long
asymmetric kernels with no dilation -- this is what may give a cleaner
receptive field on continuous angular-delay CSI.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9


__all__ = ["DCRNetV20", "FCIEncoderPath", "dcrnet_v20"]


class FCIEncoderPath(nn.Module):
    """CLNet-style replacement for ``DilateEncoderPath``.

    Pipeline:
        Conv3x3(2->2) + BN + LReLU
        Conv1x9(2->2) + BN + LReLU
        Conv9x1(2->2) + BN
        SpatialGate (3x3 over [max,mean]-pooled channels, identity-init)
        Linear(C*H*W, code_dim)
        LayerNorm (no affine) * learned scalar gate (init 1e-3)

    FLOPs vs v9u's DCREncoderBlock are roughly neutral (-7K conv, +18K SA).
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int):
        super().__init__()
        # FCI cascade: 3 convs with channel=2 throughout (CLNet encoder1)
        self.conv1 = ConvBN(in_channels, 2, 3)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = ConvBN(2, 2, [1, 9])
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3 = ConvBN(2, 2, [9, 1])

        # SpatialGate: identity-init (zero conv weight + bias=3 -> sigmoid(3)~1)
        self.use_sa = os.environ.get("DCRNET_V20_FCI_ONLY", "0") != "1"
        if self.use_sa:
            self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.sa_conv.weight)
            nn.init.constant_(self.sa_conv.bias, 3.0)

        # Same projection structure as DilateEncoderPath
        self.proj = nn.Linear(in_channels * h * w, code_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x); h = self.act1(h)
        h = self.conv2(h); h = self.act2(h)
        h = self.conv3(h)
        if self.use_sa:
            m = torch.cat([h.max(dim=1, keepdim=True)[0], h.mean(dim=1, keepdim=True)], dim=1)
            h = h * torch.sigmoid(self.sa_conv(m))
        z = self.proj(h.flatten(1))
        z = self.norm(z)
        return self.gate * z


class DCRNetV20(DCRNetV9):
    """v9 + FCI cascade enc_dilate + hsigmoid output. Decoder unchanged."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None, **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size, **kwargs)
        if self.use_dilate_path:
            if total_size is None:
                total_size = 2 * h * w
            code_dim = total_size // reduction
            self.enc_dilate = FCIEncoderPath(in_channels, h, w, code_dim)

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            if os.environ.get("DCRNET_V20_FCI_ONLY", "0") == "1":
                return refined.clamp(0.0, 1.0)
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v20(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    d_emb: int = 64,
    d_ctx: int = 32,
    refine_width: int = 2,         # v9u uses 2 (not v9 default 8)
    refine_K: int = 1,             # v9u uses 1 (not v9 default 2)
    use_dilate_path: bool = True,
    use_film: bool = False,        # v9u uses complex_encoder, not FiLM
    complex_encoder: bool = True,  # v9u
    hybrid_decoder: bool = True,   # v9u
    hybrid_direct_ranks: int = 32, # v9u
    hybrid_extra_ranks: int = 16,  # v9u
    hybrid_extra_d_emb: int = 16,  # v9u
    hybrid_gate_bias: float = -2.0,
    hybrid_alpha_init: float = 1e-2,
    enc_fuse_mlp: int = 64,        # v9u
    **kwargs,
) -> DCRNetV20:
    return DCRNetV20(
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
