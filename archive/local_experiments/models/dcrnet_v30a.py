"""DCRNet v30a: v29 + safe-only changes (centered + PE + residual local).

v30 (5 changes) underperformed v29 at same-ep — likely due to codeword split
and gate=0.05 disrupting low-rank's optimization. v30a is the SAFE subset:

3 changes from v29 (all should be free wins or near-free):
  1. z_b receives x_centered (instead of raw x) — free fix
  2. Positional embedding before transformer (~+800 params, free FLOPs)
  3. Residual on dilated local stage — free, avoids error compounding

NOT changed:
  - keep gate=1e-3 (v29 default)
  - keep single-codeword MLP fusion (no split)
  - keep ComplexLowRankEncoder output = full code_dim

This should be a near-strict improvement over v29.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV30A", "TransLitePathV30A", "dcrnet_v30a"]


class TransLitePathV30A(nn.Module):
    """v29 TransLitePathV29 + centered input + PE + residual on local stage."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        # Dilated local (multi-dilation, with residual)
        self.loc1 = ConvBN(in_channels, 2, [3, 1], dilation=1)
        self.act_l1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc2 = ConvBN(2, 2, [1, 3], dilation=2)
        self.act_l2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc3 = ConvBN(2, 2, [3, 1], dilation=3)
        self.act_l3 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        # FCI long-range
        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        self.n_tokens = h
        self.token_dim = in_channels * w
        self.bottleneck = nn.Linear(self.token_dim, d_attn)

        # Positional embedding
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_attn))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim)
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        # Keep v29 default gate
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x_centered: torch.Tensor) -> torch.Tensor:
        identity = x_centered
        h = self.loc1(x_centered); h = self.act_l1(h)
        h = self.loc2(h);            h = self.act_l2(h)
        h = self.loc3(h);            h = self.act_l3(h)
        h = h + identity                                  # residual
        h = self.long_h(h); h = self.act_lh(h)
        h = self.long_v(h)
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)
        t = t + self.pos
        t = self.transformer(t)
        t = self.token_proj(t)
        t = t.reshape(B, -1)
        z = self.global_mix(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV30A(DCRNetV9):
    """2-branch v29 with centered z_b input + PE + residual local. Single codeword."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        if self.use_dilate_path:
            self.enc_dilate = TransLitePathV30A(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_neural(x_centered)
        if self.use_dilate_path:
            z_b = self.enc_dilate(x_centered)              # centered now
            if self.enc_fuse_1x1 or self.enc_fuse_mlp_dim > 0 or self.enc_fuse_linear:
                z = self.fuse_enc(z_a, z_b)
            else:
                z = z_a + z_b
        else:
            z = z_a
        if self._use_se_z:
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v30a(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_d_attn: int = 24,
    attn_n_heads: int = 4,
    attn_ffn_hidden: int = 48,
    per_token_m: int = 8,
    **kwargs,
) -> DCRNetV30A:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV30A(
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
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        **kwargs,
    )
