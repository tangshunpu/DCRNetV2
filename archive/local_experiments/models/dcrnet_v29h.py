"""DCRNet v29h: v29 with v12r's heavy local dilated stack.

v29's TransLitePathV29 has only 3 multi-dilated convs (3x1 d=1, 1x3 d=2,
3x1 d=3) before FCI + transformer. v12r ("DCREncoderBlock") has 6 dilated
convs (full 3x1/1x3 double-stride pairs at d=1,2,3), a parallel Conv3x3
branch, concat + 1x1 fusion, and a residual skip. v12r ends up 6.77M FLOPs
vs v29 5.93M at cr=4 — the 0.83M gap is entirely in this local block.

v29h replaces the 3-layer local stage with v12r's full DCREncoderBlock,
keeping FCI + transformer + per-token + global_mix from v29 intact:

    x -> DCREncoderBlock (v12r heavy local + residual)
      -> Conv1x9 -> Conv9x1   (v29 FCI)
      -> row-token reshape -> bottleneck -> transformer -> per-token
      -> global_mix -> LN -> gate
"""
from __future__ import annotations

import torch
from torch import nn

from .dcrnet import DCREncoderBlock
from .dcrnet_v29 import DCRNetV29, TransLitePathV29
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV29H", "TransLitePathV29H", "dcrnet_v29h"]


class TransLitePathV29H(nn.Module):
    """TransLitePathV29 with DCREncoderBlock (v12r heavy local) replacing
    the 3-layer multi-dilated local stage. FCI + transformer unchanged."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        # v12r heavy local: 6-conv multi-dilated stack + parallel 3x3 + fuse + residual.
        self.local = DCREncoderBlock()

        # FCI long-range (unchanged from v29).
        from .dcrnet import ConvBN
        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        # Row-token transformer (unchanged from v29).
        self.n_tokens = h
        self.token_dim = in_channels * w

        self.bottleneck = nn.Linear(self.token_dim, d_attn)
        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim)
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.local(x)
        h = self.long_h(h); h = self.act_lh(h)
        h = self.long_v(h)
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)
        t = self.transformer(t)
        t = self.token_proj(t)
        t = t.reshape(B, -1)
        z = self.global_mix(t)
        z = self.norm(z)
        return self.gate * z


class DCRNetV29H(DCRNetV29):
    """v29 with v12r heavy local block replacing 3-layer multi-dilated stage."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
                 gate_init: float = 1e-3,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
                         attn_ffn_hidden=attn_ffn_hidden,
                         per_token_m=per_token_m, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        if self.use_dilate_path:
            self.enc_dilate = TransLitePathV29H(
                in_channels, h, w, code_dim,
                d_attn=attn_d_attn, n_heads=attn_n_heads,
                ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
            )
            with torch.no_grad():
                self.enc_dilate.gate.fill_(gate_init)


def dcrnet_v29h(
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
    gate_init: float = 1e-3,
    **kwargs,
) -> DCRNetV29H:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29H(
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
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        gate_init=gate_init,
        **kwargs,
    )
