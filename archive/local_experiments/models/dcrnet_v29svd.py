"""DCRNet v29svd: v29 with SVD-style direct LowRankDecoder.

Standard LowRankDecoder generates raw (d_k, a_k) factors for each rank,
which have scale ambiguity:  2 d_k * 0.5 a_k == d_k * a_k. The model can
absorb scale into either factor, hurting training stability.

SVDLowRankDecoder explicitly separates direction from magnitude:
  H_hat = sum_k sigma_k * u_k * v_k^H
where:
  - u_k, v_k: complex unit-norm directions
  - sigma_k: non-negative scalar (softplus)

Implementation matches the spec:
  1. direction_proj(z) -> per-rank (u_re, u_im, v_re, v_im)
  2. Normalize u, v to complex unit norm
  3. sigma_proj(z) -> softplus -> per-rank singular value
  4. Hermitian outer product with sigma scaling
  5. Sum over ranks, tanh+shift to [0, 1]

Only the *direct* branch of HybridRankDecoder is changed; the extra
GatedFactoredResidualDecoder branch (with its own gating) keeps the
v29 design.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet_v9 import DCRNetV9, HybridRankDecoder, GatedFactoredResidualDecoder
from .dcrnet_v29 import DCRNetV29, TransLitePathV29


__all__ = ["SVDLowRankDecoder", "SVDHybridRankDecoder", "DCRNetV29SVD", "dcrnet_v29svd"]


class SVDLowRankDecoder(nn.Module):
    """SVD-style rank-R decoder: H = Σ_k sigma_k * u_k * v_k^H, u/v unit-norm."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32, ranks: int = 4,
                 eps: float = 1e-8):
        super().__init__()
        self.h, self.w, self.ranks, self.eps = h, w, ranks, eps
        per_rank = 2 * h + 2 * w
        self.direction_proj = nn.Linear(code_dim, ranks * per_rank)
        self.sigma_proj = nn.Linear(code_dim, ranks)
        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h, w, r = self.h, self.w, self.ranks
        # Per-rank direction factors
        factors = self.direction_proj(z).view(b, r, 2 * h + 2 * w)
        u_re, u_im, v_re, v_im = torch.split(factors, [h, h, w, w], dim=-1)
        # Complex unit-norm normalization (per-rank)
        u_norm = (u_re.pow(2).sum(dim=-1, keepdim=True)
                  + u_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        v_norm = (v_re.pow(2).sum(dim=-1, keepdim=True)
                  + v_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        u_re = u_re / u_norm; u_im = u_im / u_norm
        v_re = v_re / v_norm; v_im = v_im / v_norm
        # Per-rank singular values (non-negative)
        sigma = F.softplus(self.sigma_proj(z))                              # (B, R)
        # Scale u by sigma (cheaper than 3-way einsum)
        sigma_u_re = sigma.unsqueeze(-1) * u_re                             # (B, R, H)
        sigma_u_im = sigma.unsqueeze(-1) * u_im
        # Hermitian outer product sum over ranks
        out_re = (torch.einsum("brh,brw->bhw", sigma_u_re, v_re)
                  + torch.einsum("brh,brw->bhw", sigma_u_im, v_im))
        out_im = (torch.einsum("brh,brw->bhw", sigma_u_im, v_re)
                  - torch.einsum("brh,brw->bhw", sigma_u_re, v_im))
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        out = torch.tanh(out)
        return (out + 1.0) * 0.5


class SVDHybridRankDecoder(nn.Module):
    """HybridRankDecoder but the direct branch uses SVDLowRankDecoder."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 direct_ranks: int = 16, extra_ranks: int = 64,
                 extra_d_emb: int = 16,
                 gate_bias: float = -2.0, alpha_init: float = 1e-2):
        super().__init__()
        self.direct = SVDLowRankDecoder(code_dim, h, w, ranks=direct_ranks)
        self.extra = GatedFactoredResidualDecoder(
            code_dim, h, w,
            extra_ranks=extra_ranks, d_emb=extra_d_emb,
            gate_bias=gate_bias, alpha_init=alpha_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.direct(z) + self.extra(z)


class DCRNetV29SVD(DCRNetV29):
    """v29 with SVD-style direct LowRankDecoder. Everything else identical."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 hybrid_direct_ranks: int = 32, hybrid_extra_ranks: int = 16,
                 hybrid_extra_d_emb: int = 16, hybrid_gate_bias: float = -2.0,
                 hybrid_alpha_init: float = 1e-2,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         hybrid_direct_ranks=hybrid_direct_ranks,
                         hybrid_extra_ranks=hybrid_extra_ranks,
                         hybrid_extra_d_emb=hybrid_extra_d_emb,
                         hybrid_gate_bias=hybrid_gate_bias,
                         hybrid_alpha_init=hybrid_alpha_init,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        # Replace the hybrid decoder with SVD variant
        self.decoder = SVDHybridRankDecoder(
            code_dim, h, w,
            direct_ranks=hybrid_direct_ranks,
            extra_ranks=hybrid_extra_ranks,
            extra_d_emb=hybrid_extra_d_emb,
            gate_bias=hybrid_gate_bias,
            alpha_init=hybrid_alpha_init,
        )


def dcrnet_v29svd(
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
    **kwargs,
) -> DCRNetV29SVD:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29SVD(
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
        **kwargs,
    )
