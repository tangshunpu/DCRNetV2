"""DCRNet v24: v20 (FCI cascade) + tiny attention branch (from v23).

v20 owns cr=4/8 on combined5 (-20.15 / -16.77) but loses cr=16 to v9u
(-13.07 vs -13.39) because dilated 7-conv cascade > FCI long kernel
when the codeword is tight. v23 showed a 16-token / 1-MHA branch added
to v9u recovers cr=16 to v9u-level. v24 combines: v20's encoder dilate
path (FCI) plus v23's attention branch, giving the union of strengths
across all cr.

Same decoder + hsigmoid as v20.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet_v20 import DCRNetV20
from .dcrnet_v23 import TinyAttnBranch


__all__ = ["DCRNetV24", "dcrnet_v24"]


class DCRNetV24(DCRNetV20):
    """v20 + parallel TinyAttnBranch encoder path."""

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
        # v20/v9 default encode flow + parallel attention branch
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
        z_c = self.attn_branch(x_centered)
        z = z + z_c
        if self._use_se_z:
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z


def dcrnet_v24(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    d_emb: int = 64,
    d_ctx: int = 32,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_patch_size: int = 8,
    attn_d_model: int = 32,
    attn_n_heads: int = 4,
    **kwargs,
) -> DCRNetV24:
    # v20 / v9u defaults
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV24(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        d_emb=d_emb, d_ctx=d_ctx,
        refine_width=refine_width, refine_K=refine_K,
        use_dilate_path=True,
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
