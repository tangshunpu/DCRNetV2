"""DCRNet v28b: v28 with larger per-token output (cr=16 push).

v28 cr=16 stalled at -12.00 @ep95, behind v9u (-12.82) and v25c (-12.48).
Diagnosis: early bottleneck Linear(64, 24) discards too much, then per-token
output (m=8) produces 256 elements compressed by Linear(256, code_dim) —
heavy info reduction across the pipeline.

v28b fix: bump per_token_m from 8 to 16. Output Linear is Linear(512, code_dim)
giving the codeword projection more headroom to compress richer attention
features.

Other components: same as v28 (FCI conv + token bottleneck d=24 + 1 transformer
+ structured per-token projection + small global mix).

FLOPs cr=16: ~395K vs v9u 380K (+4%, still ~budget).
"""
from __future__ import annotations

from .dcrnet_v28 import dcrnet_v28, DCRNetV28, TransLitePath


__all__ = ["dcrnet_v28b"]


def dcrnet_v28b(
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
    per_token_m: int = 16,        # v28b override: 8 -> 16
    **kwargs,
):
    return dcrnet_v28(
        reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        h=h, w=w, total_size=total_size,
        refine_width=refine_width, refine_K=refine_K,
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        **kwargs,
    )
