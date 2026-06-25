"""DCRNet v25: unified low-rank + hybrid dilate-FCI + slim attention.

See V20.md for the full design log. Summary:
- Keep v9u's ComplexLowRankEncoder + HybridRankDecoder core.
- enc_dilate: 1 small dilated conv (3x1, d=2) + FCI cascade (1x9 + 9x1) + SA
  -> ~3-conv block vs v9u's 7 (saves ~27K MACs).
- TinyAttnBranchSlim: 16 tokens, d_model=16, 4 heads (head_dim=4),
  zero-init projection, gated additive. Targets cr=16 attention gap (~+45K).
- Refine: FiLM-from-z (cheap, principled).
- Output: hsigmoid.

Total FLOPs ≤ v9u at all cr (saves enc_dilate convs offset by attention).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN, Channel_shuffle
from .dcrnet_v9 import DCRNetV9
from .dcrnet_v21a import DCRDecoderRefineZ, IterativeRefineZ


__all__ = ["DCRNetV25", "HybridDilateFCIEncoderPath",
           "TinyAttnBranchSlim", "dcrnet_v25"]


class HybridDilateFCIEncoderPath(nn.Module):
    """Replaces DilateEncoderPath: 1 dilated + 2 long-kernel + SA + Linear proj.

    Cascade:
        Conv3x1(d=2) + BN + LReLU       (multi-scale cheap)
        Conv1x9(d=1) + BN + LReLU       (FCI horizontal long)
        Conv9x1(d=1) + BN               (FCI vertical long)
        SpatialGate (3x3 over [max,mean] pool)
        Linear(C*H*W, code_dim) + LN + gate
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int):
        super().__init__()
        self.conv_d = ConvBN(in_channels, 2, [3, 1], dilation=2)
        self.act_d = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv_h = ConvBN(2, 2, [1, 9])
        self.act_h = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv_v = ConvBN(2, 2, [9, 1])

        # SpatialGate: identity-init
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.sa_conv.weight)
        nn.init.constant_(self.sa_conv.bias, 3.0)

        self.proj = nn.Linear(in_channels * h * w, code_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_d(x); h = self.act_d(h)
        h = self.conv_h(h); h = self.act_h(h)
        h = self.conv_v(h)
        m = torch.cat([h.max(dim=1, keepdim=True)[0], h.mean(dim=1, keepdim=True)], dim=1)
        h = h * torch.sigmoid(self.sa_conv(m))
        z = self.proj(h.flatten(1))
        z = self.norm(z)
        return self.gate * z


class TinyAttnBranchSlim(nn.Module):
    """Slim attention branch: 16 tokens / d_model=16 / 4 heads / zero-init.

    Cost at cr=4: ~45K MACs (half of v23's d_model=32 version). Targets
    cr=16 directly; at low cr its gate stays small if not useful.
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 patch_size: int = 8, d_model: int = 16, n_heads: int = 4):
        super().__init__()
        assert h % patch_size == 0 and w % patch_size == 0
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Linear(d_model, code_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x)                    # (B, d_model, 4, 4)
        t = t.flatten(2).transpose(1, 2)           # (B, 16, d_model)
        a, _ = self.attn(t, t, t, need_weights=False)
        a = a.mean(dim=1)                          # (B, d_model)
        z = self.proj(a)
        z = self.norm(z)
        return self.gate * z


class DCRNetV25(DCRNetV9):
    """v9u core + Hybrid dilate-FCI + slim attention + FiLM refine + hsigmoid."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 refine_width: int = 2, refine_K: int = 1,
                 attn_d_model: int = 16, attn_n_heads: int = 4,
                 attn_patch_size: int = 8,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         refine_width=refine_width, refine_K=refine_K,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction

        if self.use_dilate_path:
            self.enc_dilate = HybridDilateFCIEncoderPath(in_channels, h, w, code_dim)

        self.attn_branch = TinyAttnBranchSlim(
            in_channels, h, w, code_dim,
            patch_size=attn_patch_size, d_model=attn_d_model, n_heads=attn_n_heads,
        )

        # FiLM-from-z refine (cheap insurance)
        self.refine = IterativeRefineZ(width=refine_width, K=refine_K, code_dim=code_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_neural(x_centered)
        if self.use_dilate_path:
            z_b = self.enc_dilate(x)
            if self.enc_fuse_1x1 or self.enc_fuse_mlp_dim > 0 or self.enc_fuse_linear:
                z = self.fuse_enc(z_a, z_b)
            else:
                z = z_a + z_b
        else:
            z = z_a
        z_c = self.attn_branch(x_centered)
        z = z + z_c
        if self._use_se_z:
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse, z)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v25(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_d_model: int = 16,
    attn_n_heads: int = 4,
    attn_patch_size: int = 8,
    **kwargs,
) -> DCRNetV25:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV25(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        refine_width=refine_width, refine_K=refine_K,
        use_dilate_path=True,
        use_film=False, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
        enc_fuse_mlp=64,
        h=h, w=w, total_size=total_size,
        attn_d_model=attn_d_model, attn_n_heads=attn_n_heads,
        attn_patch_size=attn_patch_size,
        **kwargs,
    )
