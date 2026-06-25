"""DCRNet v30: v29 + GPT Pro-suggested upgrades (codeword split, PE, centered, residual, gate).

5 changes vs v29:
1. **Codeword budget split**: z_a = 75% code_dim (lowrank), z_b = 25% (residual),
   concatenated instead of MLP-fused. Explicit feedback-bit allocation.
2. **Positional embedding** before the transformer block (delay axis has
   physical meaning; row-tokens are not exchangeable).
3. **z_b uses x_centered** (same as z_a). Fixes the historical inconsistency
   where the dilate path received raw [0,1] input with DC bias.
4. **Residual on dilated local stage** (avoids error compounding through
   sequential d=1,2,3).
5. **Gate init 1e-3 -> 0.05** so internal supplement params get meaningful
   gradient from step 1.

Decoder unchanged from v29 (HybridRankDecoder + plain IterativeRefine + hsigmoid).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v9 import DCRNetV9, ComplexLowRankEncoder
from .dcrnet_v28 import _TransformerBlock


__all__ = ["DCRNetV30", "TransLitePathV30", "dcrnet_v30"]


class TransLitePathV30(nn.Module):
    """v29 supplement path + residual local, positional embedding, gate=0.05.

    Receives **centered** (B, 2, H, W) input. Outputs a (B, code_dim_out)
    feature vector (code_dim_out = code_b in the split-codeword setup).
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim_out: int,
                 d_attn: int = 24, n_heads: int = 4, ffn_hidden: int = 48,
                 per_token_m: int = 8):
        super().__init__()
        # Dilated local stage (with residual)
        self.loc1 = ConvBN(in_channels, 2, [3, 1], dilation=1)
        self.act_l1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc2 = ConvBN(2, 2, [1, 3], dilation=2)
        self.act_l2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.loc3 = ConvBN(2, 2, [3, 1], dilation=3)
        self.act_l3 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        # FCI long-range
        self.long_h = ConvBN(2, 2, [1, 9])
        self.act_lh = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.long_v = ConvBN(2, 2, [9, 1])

        # Token-ize
        self.n_tokens = h
        self.token_dim = in_channels * w
        self.bottleneck = nn.Linear(self.token_dim, d_attn)

        # Positional embedding (NEW)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d_attn))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.transformer = _TransformerBlock(d_attn, n_heads, ffn_hidden)

        self.token_proj = nn.Linear(d_attn, per_token_m)
        self.global_mix = nn.Linear(self.n_tokens * per_token_m, code_dim_out)
        nn.init.trunc_normal_(self.global_mix.weight, std=0.02)
        nn.init.zeros_(self.global_mix.bias)

        self.norm = nn.LayerNorm(code_dim_out, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(0.05))    # was 1e-3 in v29

    def forward(self, x_centered: torch.Tensor) -> torch.Tensor:
        # Dilated local with residual on the input
        identity = x_centered
        h = self.loc1(x_centered); h = self.act_l1(h)
        h = self.loc2(h);           h = self.act_l2(h)
        h = self.loc3(h);           h = self.act_l3(h)
        h = h + identity                                  # residual
        # FCI long-range
        h = self.long_h(h); h = self.act_lh(h)
        h = self.long_v(h)
        # Tokenize -> (B, n_tokens, token_dim)
        B, C, H, W = h.shape
        t = h.permute(0, 2, 1, 3).reshape(B, H, C * W)
        t = self.bottleneck(t)                            # (B, n_tokens, d_attn)
        t = t + self.pos                                  # positional embedding
        t = self.transformer(t)
        t = self.token_proj(t)                            # (B, n_tokens, m)
        t = t.reshape(B, -1)
        z = self.global_mix(t)                            # (B, code_dim_out)
        z = self.norm(z)
        return self.gate * z


class DCRNetV30(DCRNetV9):
    """2-branch: ComplexLowRankEncoder + TransLitePathV30 with split codeword."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 r_enc: int = 512, nonlinearity: bool = True,
                 code_a_ratio: float = 0.75,
                 attn_d_attn: int = 24, attn_n_heads: int = 4,
                 attn_ffn_hidden: int = 48, per_token_m: int = 8,
                 **kwargs):
        # Build v9u defaults via super; we'll override enc_neural / enc_dilate.
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         r_enc=r_enc, nonlinearity=nonlinearity, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        code_a = int(round(code_a_ratio * code_dim))
        code_b = code_dim - code_a
        # Make code_a / code_b multiples of 32 if possible (cleaner for downstream).
        # For cr in {4, 8, 16} default ratio 0.75 already gives (384,128), (192,64), (96,32).
        self.code_a = code_a
        self.code_b = code_b

        # Override the encoders with split codeword dims.
        self.enc_neural = ComplexLowRankEncoder(
            h, w, code_a, r_enc=r_enc, nonlinearity=nonlinearity,
        )
        self.enc_dilate = TransLitePathV30(
            in_channels, h, w, code_b,
            d_attn=attn_d_attn, n_heads=attn_n_heads,
            ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_neural(x_centered)                # (B, code_a)
        z_b = self.enc_dilate(x_centered)                # (B, code_b) — centered input!
        z = torch.cat([z_a, z_b], dim=-1)                # (B, code_dim)
        if self._use_se_z:
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v30(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    code_a_ratio: float = 0.75,
    attn_d_attn: int = 24,
    attn_n_heads: int = 4,
    attn_ffn_hidden: int = 48,
    per_token_m: int = 8,
    **kwargs,
) -> DCRNetV30:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV30(
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
        code_a_ratio=code_a_ratio,
        attn_d_attn=attn_d_attn, attn_n_heads=attn_n_heads,
        attn_ffn_hidden=attn_ffn_hidden, per_token_m=per_token_m,
        **kwargs,
    )
