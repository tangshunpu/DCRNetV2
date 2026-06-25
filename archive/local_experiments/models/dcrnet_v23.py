"""DCRNet v23: v9u + tiny transformer attention branch.

Targets the cr=16 architectural gap (v9u loses ~1.7 dB to TransNet at cr=16
on combined5). Diagnosis: bilinear + dilated conv only express separable /
local second-order correlations, attention expresses arbitrary-pair
correlations. cr=16's tight codeword (128-dim) makes the missing attention
correlations the bottleneck.

Addition: a small parallel encoder branch that patchifies H into 4x4 grid
of 8x8 patches (16 tokens, d=32), runs 1 MHA layer, mean-pools and
projects to code_dim. Gated additively into the existing fused codeword
so v9u remains functional at init (gate ~ 1e-3 -> attention contributes
nothing initially, learns to contribute).

Decoder unchanged from v9u.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet_v9 import DCRNetV9


__all__ = ["DCRNetV23", "TinyAttnBranch", "dcrnet_v23"]


class TinyAttnBranch(nn.Module):
    """Patchify -> 1 MHA layer -> mean pool -> Linear -> code_dim.

    On (B, 2, 32, 32) with patch_size=8: 16 tokens of dim d_model.
    Default d_model=32 / 4 heads / no FFN -> ~140K MACs per forward at
    cr=16 (about 10% of v9u total).
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 patch_size: int = 8, d_model: int = 32, n_heads: int = 4):
        super().__init__()
        assert h % patch_size == 0 and w % patch_size == 0
        self.n_tokens = (h // patch_size) * (w // patch_size)
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Linear(d_model, code_dim)
        # Zero-init the codeword projection so the branch contributes nothing
        # at step 0 (decoder learns from bilinear+dilate first).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 32, 32)
        t = self.patch_embed(x)                    # (B, d_model, 4, 4)
        t = t.flatten(2).transpose(1, 2)           # (B, 16, d_model)
        a, _ = self.attn(t, t, t, need_weights=False)
        a = a.mean(dim=1)                          # (B, d_model)
        z = self.proj(a)
        z = self.norm(z)
        return self.gate * z


class DCRNetV23(DCRNetV9):
    """v9u + TinyAttnBranch as a 3rd parallel encoder path."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_patch_size: int = 8, attn_d_model: int = 32,
                 attn_n_heads: int = 4,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        self.attn_branch = TinyAttnBranch(
            in_channels, h, w, code_dim,
            patch_size=attn_patch_size, d_model=attn_d_model, n_heads=attn_n_heads,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Same as v9, but add a third (attention) branch into z.
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
        # Attention branch added last; centered input (matches enc_neural).
        z_c = self.attn_branch(x_centered)
        z = z + z_c
        if self._use_se_z:
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z


def dcrnet_v23(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_patch_size: int = 8,
    attn_d_model: int = 32,
    attn_n_heads: int = 4,
    **kwargs,
) -> DCRNetV23:
    # v9u defaults: complex_encoder=True, hybrid_decoder=True, no FiLM, MLP-fuse.
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp"):
        kwargs.pop(k, None)
    return DCRNetV23(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        refine_width=refine_width, refine_K=refine_K,
        use_film=False, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
        enc_fuse_mlp=64,
        h=h, w=w, total_size=total_size,
        attn_patch_size=attn_patch_size, attn_d_model=attn_d_model,
        attn_n_heads=attn_n_heads,
        **kwargs,
    )
