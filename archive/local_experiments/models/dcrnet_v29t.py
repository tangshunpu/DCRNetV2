"""DCRNet v29t: v29 with a heavier transformer tail.

Compared to v29 (1 block, d_attn=24, ffn=48):

  - n_blocks: 1 -> 2     (deeper global mixing)
  - d_attn:   24 -> 32   (wider attention)
  - ffn_hid:  48 -> 64   (wider FFN)

Local stage (3-layer multi-dilated) and FCI (1x9 + 9x1) unchanged.
Tests whether outdoor benefits from richer non-low-rank global mixing
without touching the local convolution block.
"""
from __future__ import annotations

import torch
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v29 import DCRNetV29
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV29T", "TransLitePathV29T", "dcrnet_v29t"]


class TransLitePathV29T(nn.Module):
    """v29 dilate path with 2-block wider transformer."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 32, n_heads: int = 4, ffn_hidden: int = 64,
                 n_blocks: int = 2, per_token_m: int = 8):
        super().__init__()
        self.loc1 = ConvBN(in_channels, 2, [3, 1], dilation=1)
        self.act_l1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc2 = ConvBN(2, 2, [1, 3], dilation=2)
        self.act_l2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc3 = ConvBN(2, 2, [3, 1], dilation=3)
        self.act_l3 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        self.n_tokens = h
        self.token_dim = in_channels * w

        self.bottleneck = nn.Linear(self.token_dim, d_attn)
        self.transformer = nn.Sequential(*[
            _TransformerBlock(d_attn, n_heads, ffn_hidden) for _ in range(n_blocks)
        ])

        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim)
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.loc1(x); h = self.act_l1(h)
        h = self.loc2(h); h = self.act_l2(h)
        h = self.loc3(h); h = self.act_l3(h)
        h = self.long_h(h); h = self.act_lh(h)
        h = self.long_v(h)
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)
        t = self.transformer(t)
        t = self.token_proj(t)
        t = t.reshape(B, -1)
        z = self.global_mix(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV29T(DCRNetV29):
    """v29 with 2-block wider transformer."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_d_attn: int = 32, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 64, n_blocks: int = 2,
                 per_token_m: int = 8, **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
                         attn_ffn_hidden=attn_ffn_hidden,
                         per_token_m=per_token_m, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        if self.use_dilate_path:
            self.enc_dilate = TransLitePathV29T(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, n_blocks=n_blocks,
                per_token_m=per_token_m,
            )


def dcrnet_v29t(reduction: int = 4, expansion: int = 1, ranks: int = 16,
                r_enc: int = 512, h: int = 32, w: int = 32,
                total_size: int | None = None, refine_width: int = 2,
                refine_K: int = 1, attn_d_attn: int = 32, attn_n_heads: int = 4,
                attn_ffn_hidden: int = 64, n_blocks: int = 2,
                per_token_m: int = 8, **kwargs) -> DCRNetV29T:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29T(
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
        attn_ffn_hidden=attn_ffn_hidden, n_blocks=n_blocks,
        per_token_m=per_token_m,
        **kwargs,
    )
