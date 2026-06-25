"""DCRNet v29m: v29 with MLPs added (1) per-token after transformer, (2) inside global_mix.

Two additions vs v29's TransLitePathV29:

  (1) Per-token MLP after transformer:
        x -> bottleneck (64->24)
          -> transformer (1 block, d=24, ffn=48)
          -> MLP per-token: Linear(24->48) -> GELU -> Linear(48->24) + residual + LN
          -> token_proj (24->8)

  (2) global_mix becomes a 2-layer MLP:
        flatten (256)
        -> Linear(256->256) -> GELU
        -> Linear(256->code_dim)
        -> LN (no affine)

Local + FCI unchanged. ~+0.20M FLOPs vs v29 (cr=4).
"""
from __future__ import annotations

import torch
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v29 import DCRNetV29
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV29M", "TransLitePathV29M", "dcrnet_v29m"]


class TransLitePathV29M(nn.Module):
    """v29 dilate path with per-token MLP after transformer and 2-layer global MLP."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8, global_hidden: int = 256,
                 post_ffn_hidden: int = 48):
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
        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        # (1) Per-token MLP after transformer with residual + LN.
        self.post_norm = nn.LayerNorm(d_attn)
        self.post_ffn = nn.Sequential(
            nn.Linear(d_attn, post_ffn_hidden),
            nn.GELU(),
            nn.Linear(post_ffn_hidden, d_attn),
        )

        self.token_proj = nn.Linear(d_attn, per_token_m)

        # (2) global_mix as 2-layer MLP with GELU.
        flat_dim = self.n_tokens * per_token_m
        self.global_mix = nn.Sequential(
            nn.Linear(flat_dim, global_hidden),
            nn.GELU(),
            nn.Linear(global_hidden, code_dim),
        )
        nn.init.trunc_normal_(self.global_mix[0].weight, std=0.02)
        nn.init.zeros_(self.global_mix[0].bias)
        nn.init.trunc_normal_(self.global_mix[2].weight, std=0.02)
        nn.init.zeros_(self.global_mix[2].bias)

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
        t = t + self.post_ffn(self.post_norm(t))
        t = self.token_proj(t)
        t = t.reshape(B, -1)
        z = self.global_mix(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV29M(DCRNetV29):
    """v29 with MLP after transformer (per-token) and 2-layer MLP global_mix."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
                 global_hidden: int = 256, post_ffn_hidden: int = 48,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
                         attn_ffn_hidden=attn_ffn_hidden,
                         per_token_m=per_token_m, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        if self.use_dilate_path:
            self.enc_dilate = TransLitePathV29M(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
                global_hidden=global_hidden, post_ffn_hidden=post_ffn_hidden,
            )


def dcrnet_v29m(reduction: int = 4, expansion: int = 1, ranks: int = 16,
                r_enc: int = 512, h: int = 32, w: int = 32,
                total_size: int | None = None, refine_width: int = 2,
                refine_K: int = 1, attn_d_attn: int = 24, attn_n_heads: int = 4,
                attn_ffn_hidden: int = 48, per_token_m: int = 8,
                global_hidden: int = 256, post_ffn_hidden: int = 48,
                **kwargs) -> DCRNetV29M:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29M(
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
        global_hidden=global_hidden, post_ffn_hidden=post_ffn_hidden,
        **kwargs,
    )
