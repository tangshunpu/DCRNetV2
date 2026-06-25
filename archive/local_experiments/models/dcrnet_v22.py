"""DCRNet v22: v9u with the dilate encoder path stripped out.

Isolates the contribution of the bilinear ComplexLowRankEncoder in the v9u
configuration. v9u's encoder is a sum of (a) bilinear z_a + (b) dilate-conv
z_b through EncoderFuseMLP; v22 drops the entire dilate path (z_b, fuse).
Decoder (HybridRankDecoder + IterativeRefine + hsigmoid via DCRNET_V9_HSIGMOID
or default sigmoid) is unchanged from v9u.

Use this to measure how much the dilate path actually contributes on
combined5 at each cr (vs the bilinear path being self-sufficient).
"""
from __future__ import annotations

from .dcrnet_v9 import DCRNetV9, dcrnet_v9


__all__ = ["DCRNetV22", "dcrnet_v22"]


DCRNetV22 = DCRNetV9   # alias — same class, different default config


def dcrnet_v22(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    **kwargs,
) -> DCRNetV9:
    """v9u config with use_dilate_path=False, enc_fuse_mlp disabled."""
    # Strip dilate-related kwargs the user might pass; force off.
    for k in ("use_dilate_path", "enc_fuse_mlp", "enc_fuse_1x1", "enc_fuse_linear"):
        kwargs.pop(k, None)
    return dcrnet_v9(
        reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        refine_width=refine_width, refine_K=refine_K,
        use_film=False, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
        use_dilate_path=False,
        enc_fuse_mlp=0,
        h=h, w=w, total_size=total_size,
        **kwargs,
    )
