"""DCRNet v29mx: v29m + opened/widened hybrid decoder extra branch.

v29m fixed encoder side (MLP after transformer + 2-layer global_mix) and
showed best v29-family outdoor result so far, but still gaps v12r by
+0.16/+0.18/+0.20/0.00 dB at ep55. Diagnosis: the decoder's hybrid extra
branch (16 ranks, gate_bias=-2.0 closed) cannot express enough non-Hermitian
residual for outdoor rich scattering.

v29mx keeps v29m's encoder modifications and adds:

  hybrid_extra_ranks:  16 -> 32   (double the FiLM-modulated rank budget)
  hybrid_extra_d_emb:  16 -> 32   (wider FiLM embedding for each rank)
  hybrid_gate_bias:    -2.0 -> 0.0  (open rank-gate from start: sigmoid(0)=0.5)
  hybrid_alpha_init:   1e-2 -> 1e-1 (extra-branch weight ramps up faster)
"""
from __future__ import annotations

from .dcrnet_v29m import DCRNetV29M


__all__ = ["dcrnet_v29mx"]


def dcrnet_v29mx(reduction: int = 4, expansion: int = 1, ranks: int = 16,
                 r_enc: int = 512, h: int = 32, w: int = 32,
                 total_size: int | None = None, refine_width: int = 2,
                 refine_K: int = 1, attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
                 global_hidden: int = 256, post_ffn_hidden: int = 48,
                 **kwargs) -> DCRNetV29M:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29M(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        refine_width=refine_width, refine_K=refine_K,
        use_dilate_path=True,
        use_film=False, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=32, hybrid_extra_d_emb=32,
        hybrid_gate_bias=0.0, hybrid_alpha_init=1e-1,
        enc_fuse_mlp=64,
        h=h, w=w, total_size=total_size,
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        global_hidden=global_hidden, post_ffn_hidden=post_ffn_hidden,
        **kwargs,
    )
