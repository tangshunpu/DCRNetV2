"""DCRNet v29p: v29 with a parallel Conv3x3 + 1x1 fuse + residual in local stage.

v12r's DCREncoderBlock has a parallel Conv3x3 alongside its dilated cascade,
fused via concat + 1x1 + residual. v29 has only the 3-layer sequential
multi-dilated cascade. v29p adds the parallel 3x3 + fuse + residual on top
of v29's existing 3-layer local stage:

  seq:  loc1(3x1,d=1) -> loc2(1x3,d=2) -> loc3(3x1,d=3)  ->  s (2 ch)
  par:  Conv3x3(d=1)                                     ->  p (2 ch)
  fuse: concat([s, p]) -> 1x1 ConvBN -> (2 ch)
  res:  return x + fused

FCI + transformer + global_mix unchanged. Tests whether v12r-style
multi-scale local + residual fixes the slow start on outdoor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v29 import DCRNetV29
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV29P", "TransLitePathV29P", "dcrnet_v29p"]


class TransLitePathV29P(nn.Module):
    """v29 dilate path with parallel Conv3x3 + 1x1 fuse + residual in local stage."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        self.loc1 = ConvBN(in_channels, 2, [3, 1], dilation=1)
        self.act_l1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc2 = ConvBN(2, 2, [1, 3], dilation=2)
        self.act_l2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc3 = ConvBN(2, 2, [3, 1], dilation=3)
        self.act_l3 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.par = ConvBN(in_channels, 2, 3, dilation=1)
        self.act_par = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.fuse = ConvBN(4, 2, 1)
        self.act_res = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        self.n_tokens = h
        self.token_dim = in_channels * w

        self.bottleneck = nn.Linear(self.token_dim, d_attn)
        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim)
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.loc1(x); s = self.act_l1(s)
        s = self.loc2(s); s = self.act_l2(s)
        s = self.loc3(s); s = self.act_l3(s)
        p = self.par(x); p = self.act_par(p)
        fused = self.fuse(torch.cat([s, p], dim=1))
        h = self.act_res(x + fused)

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


class DCRNetV29P(DCRNetV29):
    """v29 with parallel 3x3 + residual in local stage."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
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
            self.enc_dilate = TransLitePathV29P(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
            )


def dcrnet_v29p(reduction: int = 4, expansion: int = 1, ranks: int = 16,
                r_enc: int = 512, h: int = 32, w: int = 32,
                total_size: int | None = None, refine_width: int = 2,
                refine_K: int = 1, attn_d_attn: int = 24, attn_n_heads: int = 4,
                attn_ffn_hidden: int = 48, per_token_m: int = 8,
                **kwargs) -> DCRNetV29P:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29P(
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
