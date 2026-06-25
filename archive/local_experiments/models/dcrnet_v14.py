"""DCRNet v14: Bilinear + PatchTransformer + RankQueryTransformerDecoder.

Replaces v13's HybridRankDecoder with a Rank-Query Transformer Decoder:
R learnable rank queries cross-modulate with z (FiLM) then self-attend
to coordinate. Each query emits (d, a) complex factors → rank-1 outer
product. Decoder = Σ_r d_r ⊗ a_r^H.

Architecture:
    Encoder = ComplexBilinear + PatchTransformer (same as v13)
    Decoder = RankQueryTransformer (NEW)
    Refine  = IterativeRefine

Compared to v13 HybridDecoder:
    - v13: direct LowRankDecoder + cheap factored extras (z gating)
    - v14: R rank queries attend each other + FiLM(z) modulation
    → v14 ranks "talk to each other" via self-attention, can specialize

FLOPs cr=4 v14-mid (R=24, dim=64, depth=2): ~8M
"""

from __future__ import annotations
import torch
from torch import nn

from .dcrnet_v9 import (
    ComplexLowRankEncoder,
    EncoderFuseMLP,
    IterativeRefine,
)
from .dcrnet_v13 import PatchTransformerEncoder


__all__ = ["DCRNetV14", "RankQueryTransformerDecoder", "dcrnet_v14"]


class RankQueryTransformerDecoder(nn.Module):
    """R learnable rank queries → FiLM(z) → self-attention → per-rank factor heads.

    Each rank query owns a (dim)-d embedding that produces one complex rank-1
    outer product (d_r ⊗ a_r^H). Queries self-attend across ranks for
    coordination (so different ranks specialize to different multipath
    components rather than duplicating).

    FiLM init is zero so γ=1+0=1, β=0 → decoder starts as a static rank-bank
    (z has no effect). Adaptation is learned gradually.
    """

    def __init__(
        self,
        code_dim: int,
        h: int = 32,
        w: int = 32,
        ranks: int = 24,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.h, self.w = h, w
        self.ranks = ranks
        self.dim = dim

        # Learnable rank queries (each rank owns dim-d embedding)
        self.rank_queries = nn.Parameter(torch.empty(1, ranks, dim))
        nn.init.trunc_normal_(self.rank_queries, std=0.02)

        # FiLM modulation: z → γ, β. Zero-init: γ=1+0, β=0 → static at start.
        self.z_film = nn.Linear(code_dim, 2 * dim)
        nn.init.zeros_(self.z_film.weight)
        nn.init.zeros_(self.z_film.bias)

        # Self-attention blocks over rank queries
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, heads, batch_first=True),
                "ln2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * ffn_mult),
                    nn.GELU(),
                    nn.Linear(dim * ffn_mult, dim),
                ),
            }))

        # Per-rank factor heads
        self.factor_d_re = nn.Linear(dim, h)
        self.factor_d_im = nn.Linear(dim, h)
        self.factor_a_re = nn.Linear(dim, w)
        self.factor_a_im = nn.Linear(dim, w)

        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, code_dim)
        B = z.shape[0]
        # FiLM modulation
        film = self.z_film(z)  # (B, 2*dim), zero-init at step 0
        gamma, beta = film.chunk(2, dim=-1)  # each (B, dim)
        # Apply to rank queries: shape (B, R, dim)
        queries = (1.0 + gamma.unsqueeze(1)) * self.rank_queries + beta.unsqueeze(1)

        # Self-attention
        for blk in self.blocks:
            h_in = blk["ln1"](queries)
            attn_out, _ = blk["attn"](h_in, h_in, h_in, need_weights=False)
            queries = queries + attn_out
            queries = queries + blk["ffn"](blk["ln2"](queries))

        # Per-rank factor extraction
        d_re = self.factor_d_re(queries)  # (B, R, h)
        d_im = self.factor_d_im(queries)
        a_re = self.factor_a_re(queries)
        a_im = self.factor_a_im(queries)

        # Complex outer products: (d_re + i d_im) ⊗ (a_re + i a_im)^*
        out_re = (torch.einsum("brh,brw->bhw", d_re, a_re)
                  + torch.einsum("brh,brw->bhw", d_im, a_im))
        out_im = (torch.einsum("brh,brw->bhw", d_im, a_re)
                  - torch.einsum("brh,brw->bhw", d_re, a_im))
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        out = torch.tanh(out)
        return (out + 1.0) * 0.5


class DCRNetV14(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        expansion: int = 1,
        # bilinear encoder
        r_enc: int = 384,
        # patch transformer encoder
        patch_size: int = 8,
        patch_dim: int = 96,
        patch_depth: int = 2,
        patch_heads: int = 4,
        patch_ffn_mult: int = 2,
        # rank-query transformer decoder
        dec_ranks: int = 24,
        dec_dim: int = 64,
        dec_depth: int = 2,
        dec_heads: int = 4,
        dec_ffn_mult: int = 2,
        # refine
        refine_width: int = 2,
        refine_K: int = 1,
        # encoder fuse
        enc_fuse_mlp: int = 64,
    ):
        super().__init__()
        h, w = 32, 32
        code_dim = 2048 // reduction
        del expansion

        self.enc_bilinear = ComplexLowRankEncoder(h, w, code_dim, r_enc=r_enc)
        self.enc_patch = PatchTransformerEncoder(
            in_channels, h, w, code_dim,
            patch_size=patch_size,
            dim=patch_dim,
            depth=patch_depth,
            heads=patch_heads,
            ffn_mult=patch_ffn_mult,
        )
        self.fuse_enc = EncoderFuseMLP(code_dim, hidden=enc_fuse_mlp)

        self.decoder = RankQueryTransformerDecoder(
            code_dim, h, w,
            ranks=dec_ranks,
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            ffn_mult=dec_ffn_mult,
        )
        self.refine = IterativeRefine(width=refine_width, K=refine_K)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_bilinear(x_centered)
        z_b = self.enc_patch(x)
        return self.fuse_enc(z_a, z_b)

    def _coarse_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        return refined.clamp(0.0, 1.0) if clamp else refined

    def decode_with_coarse(self, z: torch.Tensor, clamp: bool = True):
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        if clamp:
            return refined.clamp(0.0, 1.0), coarse.clamp(0.0, 1.0)
        return refined, coarse

    def forward(self, x: torch.Tensor, return_coarse: bool = False, clamp: bool = True):
        z = self.encode(x)
        if return_coarse:
            return self.decode_with_coarse(z, clamp=clamp)
        return self.decode(z, clamp=clamp)


def dcrnet_v14(
    reduction: int = 4,
    expansion: int = 1,
    r_enc: int = 384,
    patch_size: int = 8,
    patch_dim: int = 96,
    patch_depth: int = 2,
    patch_heads: int = 4,
    patch_ffn_mult: int = 2,
    dec_ranks: int = 24,
    dec_dim: int = 64,
    dec_depth: int = 2,
    dec_heads: int = 4,
    dec_ffn_mult: int = 2,
    refine_width: int = 2,
    refine_K: int = 1,
    enc_fuse_mlp: int = 64,
    **kwargs,
) -> DCRNetV14:
    return DCRNetV14(
        reduction=reduction,
        expansion=expansion,
        r_enc=r_enc,
        patch_size=patch_size,
        patch_dim=patch_dim,
        patch_depth=patch_depth,
        patch_heads=patch_heads,
        patch_ffn_mult=patch_ffn_mult,
        dec_ranks=dec_ranks,
        dec_dim=dec_dim,
        dec_depth=dec_depth,
        dec_heads=dec_heads,
        dec_ffn_mult=dec_ffn_mult,
        refine_width=refine_width,
        refine_K=refine_K,
        enc_fuse_mlp=enc_fuse_mlp,
    )
