"""DCRNet v28: TransNet- and CLNet-inspired clean redesign of the dilate path.

Key diagnosis of v9u DilateEncoderPath:
  - 7 small (2->2) conv layers extract VERY limited features (each layer
    has only 36 params; in total ~250 params for conv block).
  - Linear(2*H*W, code_dim) is the dominant cost (~1M MACs at cr=4).
  - The weak conv stack + heavy Linear is *imbalanced*: the conv barely
    enriches features before being asked to project through a dense
    bottleneck.

TransNet succeeds because its dense Linear(2048, code_dim) comes after
2x Transformer layers (d=64) that fully mix all positions globally —
so the Linear is "extracting" from already-mixed features.

CLNet succeeds because it has two branches (long-kernel + channel
expansion) producing rich features before Conv1d compression.

v28's z_b supplement combines both insights with a *structured* output
projection (per-token + small global mix), avoiding the heavy
Linear(2048, code_dim) bottleneck.

ENCODER (2 branches):
  z_a = ComplexLowRankEncoder(x)                          [low-rank prior]
  z_b = TransLitePath(x):
        Conv3x3 + Conv1x9 + Conv9x1  (CLNet FCI cascade)  [local long-range]
        Reshape -> (B, 32 row-tokens, 64 dim)             [TransNet token-view]
        Linear(64, d_attn=24)                             [token bottleneck]
        1 Transformer block (d=24, h=4, FFN=48)           [TransNet global mix]
        per-token Linear(24, 8) -> (B, 256)               [structured]
        Linear(256, code_dim)                             [small global mix]
        LN + gate
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9


__all__ = ["DCRNetV28", "TransLitePath", "dcrnet_v28"]


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, t):
        n = self.norm1(t)
        a, _ = self.attn(n, n, n, need_weights=False)
        t = t + a
        t = t + self.ffn(self.norm2(t))
        return t


class TransLitePath(nn.Module):
    """Lightweight TransNet+CLNet-inspired supplement to ComplexLowRankEncoder.

    Pipeline (32x32 input):
      Conv3x3(2->2) + LReLU   [CLNet FCI: small local]
      Conv1x9(2->2) + LReLU   [CLNet FCI: long horizontal]
      Conv9x1(2->2)           [CLNet FCI: long vertical]
      Reshape  (B, 2, 32, 32) -> (B, 32 row-tokens, 64 dim)
      Linear(64, d_attn) bottleneck
      1x Transformer block    [TransNet-style global mix]
      Linear(d_attn, m) per token, m=8
      Reshape (B, 32, 8) -> (B, 256)
      Linear(256, code_dim)
      LN + gate (init 1e-3)
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        # FCI cascade (CLNet encoder1 style)
        self.conv1 = ConvBN(in_channels, 2, 3)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv2 = ConvBN(2, 2, [1, 9])
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv3 = ConvBN(2, 2, [9, 1])

        # Token-ize: (B, 2, h, w) -> (B, h, 2*w) row-tokens
        self.n_tokens = h
        self.token_dim = in_channels * w   # 2*32 = 64

        self.bottleneck = nn.Linear(self.token_dim, d_attn)
        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        # Structured output: per-token projection + small global mix
        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim)
        # Zero-init the global_mix output bias and small init on weights
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x); h = self.act1(h)
        h = self.conv2(h); h = self.act2(h)
        h = self.conv3(h)
        # (B, 2, 32, 32) -> (B, 32 row-tokens, 2*32 dim)
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)               # (B, 32, d_attn)
        t = self.transformer(t)              # (B, 32, d_attn)
        t = self.token_proj(t)               # (B, 32, m)
        t = t.reshape(B, -1)                 # (B, 32*m)
        z = self.global_mix(t)               # (B, code_dim)
        z = self.norm(z)
        return self.gate * z


class DCRNetV28(DCRNetV9):
    """2-branch: ComplexLowRankEncoder + TransLitePath."""

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
            self.enc_dilate = TransLitePath(
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


def dcrnet_v28(
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
) -> DCRNetV28:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV28(
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
