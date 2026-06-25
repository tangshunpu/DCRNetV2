"""DCRNet v26: clean 2-branch design — low-rank + (dilated then attention).

Returns to the fundamental low-rank-prior philosophy. Drops v25*'s
"v9u backbone + attention side branch" stack (3 effective paths). Instead:

ENCODER (2 branches):
    z_a = ComplexLowRankEncoder(x)      [main: low-rank Hermitian bilinear]
    z_b = DilatedAttnPath(x)            [supplement: dilated -> patch -> attn]
    z   = fuse_mlp(z_a, z_b)            [v9u-style additive-MLP fusion]

DilatedAttnPath replaces v9u's entire DCREncoderBlock + Linear(2048, D)
with a structured sequential module:
    Conv3x1(d=2) -> LReLU -> Conv1x3(d=3) -> LReLU  (multi-scale local)
    Conv3x3(d=1)                                    (small fuse)
    PatchEmbed(2 -> d_model, patch=8)               (4x4=16 tokens, d=24)
    1x Transformer block (MHA + FFN)                (global pairs)
    mean-pool over tokens
    Linear(d_model, code_dim)                       (cheap project to z_b)
    LN + learned scalar gate (init 1e-3)

By replacing the dense Linear(2048, code_dim) with PatchEmbed + small
attention + Linear(d_model, code_dim), we *save* hundreds of K of MACs vs
v9u's dilate path, then spend a fraction on attention. Net FLOPs < v9u
at all cr.

DECODER:
    HybridRankDecoder (v9u)             [low-rank prior decoder]
    IterativeRefine (v8/v9u plain — no FiLM)
    hsigmoid output
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9


__all__ = ["DCRNetV26", "DilatedAttnPath", "dcrnet_v26"]


class _TransformerBlock(nn.Module):
    """Pre-LN MHA + FFN."""
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


class DilatedAttnPath(nn.Module):
    """Supplement branch: dilated conv -> patchify -> transformer -> proj.

    Captures (a) multi-scale local correlations via dilated convs (b)
    arbitrary-pair global correlations via 1 attention layer. Output is
    a code_dim vector gated additively into the bilinear main codeword.
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 patch_size: int = 8, d_model: int = 24, n_heads: int = 4,
                 ffn_hidden: int = 48):
        super().__init__()
        # Dilated multi-scale conv block (channel=2 throughout for cheap)
        self.conv_d1 = ConvBN(in_channels, 2, [3, 1], dilation=2)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv_d2 = ConvBN(2, 2, [1, 3], dilation=3)
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.conv_d3 = ConvBN(2, 2, 3, dilation=1)         # small fuse
        # PatchEmbed: 2-ch -> d_model tokens (8x8 patches -> 16 tokens)
        assert h % patch_size == 0 and w % patch_size == 0
        self.patch_embed = nn.Conv2d(2, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.block = _TransformerBlock(d_model, n_heads, ffn_hidden)
        self.proj = nn.Linear(d_model, code_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_d1(x); h = self.act1(h)
        h = self.conv_d2(h); h = self.act2(h)
        h = self.conv_d3(h)
        t = self.patch_embed(h).flatten(2).transpose(1, 2)   # (B, 16, d_model)
        t = self.block(t)
        t = t.mean(dim=1)                                     # (B, d_model)
        z = self.proj(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV26(DCRNetV9):
    """2-branch encoder: ComplexLowRankEncoder + DilatedAttnPath."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_patch_size: int = 8, attn_d_model: int = 24,
                 attn_n_heads: int = 4, attn_ffn_hidden: int = 48,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        # Replace the dilate encoder path entirely with DilatedAttnPath
        if self.use_dilate_path:
            self.enc_dilate = DilatedAttnPath(
                in_channels, h, w, code_dim,
                patch_size=attn_patch_size, d_model=attn_d_model,
                n_heads=attn_n_heads, ffn_hidden=attn_ffn_hidden,
            )

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v26(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_d_model: int = 24,
    attn_n_heads: int = 4,
    attn_patch_size: int = 8,
    attn_ffn_hidden: int = 48,
    **kwargs,
) -> DCRNetV26:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV26(
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
        attn_patch_size=attn_patch_size, attn_ffn_hidden=attn_ffn_hidden,
        **kwargs,
    )
