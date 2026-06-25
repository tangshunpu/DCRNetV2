"""DCRNet v13: ComplexBilinear + PatchTransformer + HybridDecoder.

TransNet-inspired: replaces v9's DilateEncoderPath (DCREncoderBlock +
Linear 2048->D) with a patch-based transformer encoder. Keeps all v9
innovations: ComplexLowRankEncoder, HybridRankDecoder, IterativeRefine.

Architecture:
    Input H (B,2,32,32)
        │
   ┌────┴────────┐
   │             │
   ▼             ▼
ComplexLowRank   PatchTransformer
 (bilinear)      (8x8 patches, 16 tokens, MHA + FFN)
   │             │
   ▼             ▼
   z_a           z_b
        │
        ▼
   EncoderFuseMLP (concat + MLP residual)
        │
        ▼
   HybridRankDecoder (direct + cheap extras)
        │
        ▼ IterativeRefine
   H_recon

Default v13-mid config (cr=4):
    r_enc=384, patch_dim=96, depth=2 → ~7.5M FLOPs
"""

from __future__ import annotations
import torch
from torch import nn

from .dcrnet_v9 import (
    ComplexLowRankEncoder,
    EncoderFuseMLP,
    HybridRankDecoder,
    IterativeRefine,
)


__all__ = ["DCRNetV13", "PatchTransformerEncoder", "dcrnet_v13"]


class PatchTransformerEncoder(nn.Module):
    """TransNet-style patch encoder, lightweight (16 tokens at 8x8 patches).

    For 32x32 input, 8x8 patches gives 16 tokens. Each block is pre-LN
    MHA + FFN. Output projection mapped to code_dim, zero-init so initial
    contribution is zero (model starts using bilinear only, learns
    transformer gradually).
    """

    def __init__(
        self,
        in_channels: int = 2,
        h: int = 32,
        w: int = 32,
        code_dim: int = 512,
        patch_size: int = 8,
        dim: int = 96,
        depth: int = 2,
        heads: int = 4,
        ffn_mult: int = 2,
    ):
        super().__init__()
        assert h % patch_size == 0 and w % patch_size == 0
        self.in_channels = in_channels
        self.h, self.w = h, w
        self.patch_size = patch_size
        self.dim = dim
        self.n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        self.proj = nn.Linear(self.n_patches * dim, code_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B = x.shape[0]
        ps = self.patch_size
        nph, npw = self.h // ps, self.w // ps
        # patchify (B, C, H, W) -> (B, N, C*ps*ps)
        patches = x.reshape(B, self.in_channels, nph, ps, npw, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B, self.n_patches, -1)

        z = self.patch_embed(patches) + self.pos_embed
        for blk in self.blocks:
            h_in = blk["ln1"](z)
            attn_out, _ = blk["attn"](h_in, h_in, h_in, need_weights=False)
            z = z + attn_out
            z = z + blk["ffn"](blk["ln2"](z))

        return self.proj(z.reshape(B, -1))


class DCRNetV13(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        expansion: int = 1,
        # bilinear encoder
        r_enc: int = 384,
        # patch transformer
        patch_size: int = 8,
        patch_dim: int = 96,
        patch_depth: int = 2,
        patch_heads: int = 4,
        patch_ffn_mult: int = 2,
        # hybrid decoder
        direct_ranks: int = 20,
        extra_ranks: int = 16,
        extra_d_emb: int = 8,
        gate_bias: float = -2.0,
        alpha_init: float = 0.05,
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

        self.decoder = HybridRankDecoder(
            code_dim, h, w,
            direct_ranks=direct_ranks,
            extra_ranks=extra_ranks,
            extra_d_emb=extra_d_emb,
            gate_bias=gate_bias,
            alpha_init=alpha_init,
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


def dcrnet_v13(
    reduction: int = 4,
    expansion: int = 1,
    r_enc: int = 384,
    patch_size: int = 8,
    patch_dim: int = 96,
    patch_depth: int = 2,
    patch_heads: int = 4,
    patch_ffn_mult: int = 2,
    direct_ranks: int = 20,
    extra_ranks: int = 16,
    extra_d_emb: int = 8,
    gate_bias: float = -2.0,
    alpha_init: float = 0.05,
    refine_width: int = 2,
    refine_K: int = 1,
    enc_fuse_mlp: int = 64,
    **kwargs,
) -> DCRNetV13:
    return DCRNetV13(
        reduction=reduction,
        expansion=expansion,
        r_enc=r_enc,
        patch_size=patch_size,
        patch_dim=patch_dim,
        patch_depth=patch_depth,
        patch_heads=patch_heads,
        patch_ffn_mult=patch_ffn_mult,
        direct_ranks=direct_ranks,
        extra_ranks=extra_ranks,
        extra_d_emb=extra_d_emb,
        gate_bias=gate_bias,
        alpha_init=alpha_init,
        refine_width=refine_width,
        refine_K=refine_K,
        enc_fuse_mlp=enc_fuse_mlp,
    )
