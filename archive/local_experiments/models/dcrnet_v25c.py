"""DCRNet v25c: v9u (unchanged encoder) + medium attention + FiLM refine + hsigmoid.

v25 (HybridDilateFCI) was too lean — simplified v9u's 7-conv dilated cascade
to 3 convs and used d_model=16 slim attention. Result: cr=16 stuck at -12.2
even @ep210, slower than v9u/v23 same-ep. Diagnosis: v22 already showed the
dilate path matters; trimming it costs cr=4 (low-frequency reduction).

v25c reverts to v9u's full encoder backbone (DCREncoderBlock untouched),
upgrades attention to d_model=24 + FFN (like v25b), keeps FiLM-z refine
and hsigmoid. This is essentially v23 with the cheap decoder upgrades.

FLOPs: v9u + ~55K (within ~+1% of v9u).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet_v9 import DCRNetV9
from .dcrnet_v21a import IterativeRefineZ
from .dcrnet_v25b import TinyAttnBranchPlus


__all__ = ["DCRNetV25C", "dcrnet_v25c"]


class DCRNetV25C(DCRNetV9):
    """v9u + TinyAttnBranchPlus + FiLM refine + hsigmoid. Encoder backbone untouched."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 refine_width: int = 2, refine_K: int = 1,
                 attn_d_model: int = 24, attn_n_heads: int = 4,
                 attn_patch_size: int = 8, attn_ffn_hidden: int = 48,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         refine_width=refine_width, refine_K=refine_K,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction

        # enc_dilate untouched (v9u DCREncoderBlock)
        self.attn_branch = TinyAttnBranchPlus(
            in_channels, h, w, code_dim,
            patch_size=attn_patch_size, d_model=attn_d_model, n_heads=attn_n_heads,
            ffn_hidden=attn_ffn_hidden,
        )
        self.refine = IterativeRefineZ(width=refine_width, K=refine_K, code_dim=code_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
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

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse, z)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v25c(
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
) -> DCRNetV25C:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV25C(
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
