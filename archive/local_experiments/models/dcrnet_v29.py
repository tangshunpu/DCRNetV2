"""DCRNet v29: v28 with dilated-local stage in place of plain Conv3x3.

v28 supplement (TransLitePath) used Conv3x3 + Conv1x9 + Conv9x1 + transformer.
Diagnosis: the only "local" capture was a single Conv3x3(d=1); long kernels
+ transformer dominate, leaving sparse local detail (peaks, sharp edges)
under-represented.

v29 replaces the leading Conv3x3 with a 3-conv MULTI-DILATED stack
(dilation 1, 2, 3) — same total FLOPs but multi-scale local receptive
fields (3x1, then 1x5-with-holes via d=2 on 1x3, then 7x1-with-holes via
d=3 on 3x1). This is essentially v9u's DCREncoderBlock dilation idea
condensed to 3 layers, integrated WITH the FCI long-range + transformer
global mixing.

ENCODER (still 2 branches):
  z_a = ComplexLowRankEncoder(x)               [low-rank prior]
  z_b = TransLitePathV29(x):
      Multi-dilated local:
        Conv3x1(d=1) + LReLU
        Conv1x3(d=2) + LReLU
        Conv3x1(d=3) + LReLU
      Long-range FCI:
        Conv1x9(d=1) + LReLU
        Conv9x1(d=1)
      Reshape -> (B, 32 tokens, 64 dim)
      Linear bottleneck -> d_attn=24
      1 Transformer block (h=4, FFN=48)
      per-token Linear(24, m=8)
      Linear(256, code_dim)
      LN + gate

FLOPs vs v28: equal (3 dilated 3x1/1x3 ≈ 1 Conv3x3 by FLOPs).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV29", "TransLitePathV29", "dcrnet_v29"]


class TransLitePathV29(nn.Module):
    """v28's TransLitePath with multi-dilated local stage (replaces Conv3x3)."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        # Multi-dilated local (replaces v28 Conv3x3 leading layer)
        self.loc1 = ConvBN(in_channels, 2, [3, 1], dilation=1)
        self.act_l1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc2 = ConvBN(2, 2, [1, 3], dilation=2)
        self.act_l2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc3 = ConvBN(2, 2, [3, 1], dilation=3)
        self.act_l3 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        # FCI long-range (unchanged from v28)
        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        # Token-ize: (B, 2, h, w) -> (B, h, 2*w) row-tokens
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
        h = self.loc1(x); h = self.act_l1(h)
        h = self.loc2(h); h = self.act_l2(h)
        h = self.loc3(h); h = self.act_l3(h)
        h = self.long_h(h); h = self.act_lh(h)
        h = self.long_v(h)
        # (B, 2, 32, 32) -> (B, 32, 64) row-tokens
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)
        t = self.transformer(t)
        t = self.token_proj(t)
        t = t.reshape(B, -1)
        z = self.global_mix(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV29(DCRNetV9):
    """2-branch: ComplexLowRankEncoder + TransLitePathV29 (dilated-local + FCI + attn)."""

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
            self.enc_dilate = TransLitePathV29(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
            )

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v29(
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
) -> DCRNetV29:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29(
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
