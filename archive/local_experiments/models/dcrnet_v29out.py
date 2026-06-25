"""DCRNet v29out: v29 outdoor-tuned variant.

Two minimal hyperparameter changes vs v29 (no structural change):

  1. dilate-branch gate init: 1e-3 -> 0.1
       v29 z_b path is almost silenced at start (gate~0), so it takes many
       hundreds of epochs to warm up. On outdoor, where multi-path is rich
       and the low-rank Hermitian prior cannot capture everything, the
       transformer / FCI branch must contribute early.

  2. hybrid extra-branch gate bias: -2.0 -> 0.0
       v29 starts the FiLM-modulated extra ranks at sigmoid(-2)~=0.12 (mostly
       closed). On outdoor, open them from the start (sigmoid(0)=0.5).

  hybrid_extra_ranks stays at 16 (no extra capacity added).
"""
from __future__ import annotations

import torch
from torch import nn

from .dcrnet_v29 import DCRNetV29, TransLitePathV29


__all__ = ["DCRNetV29Out", "dcrnet_v29out"]


class DCRNetV29Out(DCRNetV29):
    """v29 with z_b gate_init and hybrid_gate_bias raised for outdoor."""

    def __init__(self, *args, gate_init: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_dilate_path:
            assert isinstance(self.enc_dilate, TransLitePathV29)
            with torch.no_grad():
                self.enc_dilate.gate.fill_(gate_init)


def dcrnet_v29out(
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
    gate_init: float = 0.1,
    **kwargs,
) -> DCRNetV29Out:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29Out(
        in_channels=2, reduction=reduction, expansion=expansion,
        ranks=ranks, r_enc=r_enc,
        refine_width=refine_width, refine_K=refine_K,
        use_dilate_path=True,
        use_film=False, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=0.0, hybrid_alpha_init=1e-2,
        enc_fuse_mlp=64,
        h=h, w=w, total_size=total_size,
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        gate_init=gate_init,
        **kwargs,
    )
