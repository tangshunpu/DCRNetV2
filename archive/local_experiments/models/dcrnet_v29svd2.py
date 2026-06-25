"""DCRNet v29svd2: v29svd + sigma_bias schedule + SVD parameterization on encoder.

Two additions over v29svd:

1. **sigma_bias_init schedule** on decoder direct branch:
   sigma_bias linearly anneals from sigma_bias_start (+2.0) at epoch 0
   to sigma_bias_end (-2.0) at the final epoch. Early on softplus(.+2)~2
   keeps all ranks active (dense regime, fast convergence); late on
   softplus(.-2)~0.13 forces sparsity (only large signals matter).

2. **SVD parameterization on the encoder** (SVDComplexLowRankEncoder):
   The complex bilinear u, v are normalized to unit norm per rank, and a
   per-rank sigma scalar (with the same schedule mechanism) carries scale.
   Removes scale ambiguity on the encoder side too.

Schedule is updated each epoch via `model.update_sigma_bias(epoch, total_epochs)`
called from the training loop in main.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet_v9 import DCRNetV9, GatedFactoredResidualDecoder
from .dcrnet_v29 import DCRNetV29


__all__ = ["DCRNetV29SVD2", "SVDLowRankDecoderSched",
           "SVDComplexLowRankEncoder", "SVDHybridRankDecoder2", "dcrnet_v29svd2"]


class SVDLowRankDecoderSched(nn.Module):
    """SVD decoder with sigma_bias schedule.

    sigma = softplus(sigma_proj(z) + current_sigma_bias)
    current_sigma_bias is set externally via update_sigma_bias(epoch, total).
    """

    def __init__(self, code_dim: int, h: int = 32, w: int = 32, ranks: int = 4,
                 sigma_bias_start: float = 2.0, sigma_bias_end: float = -2.0,
                 eps: float = 1e-8):
        super().__init__()
        self.h, self.w, self.ranks, self.eps = h, w, ranks, eps
        per_rank = 2 * h + 2 * w
        self.direction_proj = nn.Linear(code_dim, ranks * per_rank)
        self.sigma_proj = nn.Linear(code_dim, ranks)
        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))
        self.sigma_bias_start = sigma_bias_start
        self.sigma_bias_end = sigma_bias_end
        self.register_buffer("current_sigma_bias",
                             torch.tensor(sigma_bias_start, dtype=torch.float32))

    def update_sigma_bias(self, epoch: int, total_epochs: int) -> None:
        progress = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
        val = self.sigma_bias_start * (1.0 - progress) + self.sigma_bias_end * progress
        self.current_sigma_bias.fill_(val)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h, w, r = self.h, self.w, self.ranks
        factors = self.direction_proj(z).view(b, r, 2 * h + 2 * w)
        u_re, u_im, v_re, v_im = torch.split(factors, [h, h, w, w], dim=-1)
        u_norm = (u_re.pow(2).sum(dim=-1, keepdim=True)
                  + u_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        v_norm = (v_re.pow(2).sum(dim=-1, keepdim=True)
                  + v_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        u_re = u_re / u_norm; u_im = u_im / u_norm
        v_re = v_re / v_norm; v_im = v_im / v_norm
        sigma = F.softplus(self.sigma_proj(z) + self.current_sigma_bias)         # (B, R)
        sigma_u_re = sigma.unsqueeze(-1) * u_re
        sigma_u_im = sigma.unsqueeze(-1) * u_im
        out_re = (torch.einsum("brh,brw->bhw", sigma_u_re, v_re)
                  + torch.einsum("brh,brw->bhw", sigma_u_im, v_im))
        out_im = (torch.einsum("brh,brw->bhw", sigma_u_im, v_re)
                  - torch.einsum("brh,brw->bhw", sigma_u_re, v_im))
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        out = torch.tanh(out)
        return (out + 1.0) * 0.5


class SVDComplexLowRankEncoder(nn.Module):
    """ComplexLowRankEncoder with SVD parameterization: unit-norm u, v +
    per-rank sigma (softplus(raw + current_sigma_bias)).

    Removes scale ambiguity on the encoder side: u, v can no longer
    collapse to 0 or blow up; only sigma carries magnitude.

    Same forward math as ComplexLowRankEncoder otherwise.
    """

    def __init__(self, h: int, w: int, code_dim: int, r_enc: int = 512,
                 nonlinearity: bool = True,
                 sigma_bias_start: float = 2.0, sigma_bias_end: float = -2.0,
                 eps: float = 1e-8):
        super().__init__()
        self.r_enc, self.h, self.w, self.eps = r_enc, h, w, eps
        self.u_re = nn.Parameter(torch.empty(r_enc, h))
        self.u_im = nn.Parameter(torch.empty(r_enc, h))
        self.v_re = nn.Parameter(torch.empty(r_enc, w))
        self.v_im = nn.Parameter(torch.empty(r_enc, w))
        nn.init.trunc_normal_(self.u_re, std=h ** -0.5)
        nn.init.trunc_normal_(self.u_im, std=h ** -0.5)
        nn.init.trunc_normal_(self.v_re, std=w ** -0.5)
        nn.init.trunc_normal_(self.v_im, std=w ** -0.5)
        # Per-rank sigma (raw param, gets softplus(raw + bias))
        self.sigma_raw = nn.Parameter(torch.zeros(r_enc))
        self.sigma_bias_start = sigma_bias_start
        self.sigma_bias_end = sigma_bias_end
        self.register_buffer("current_sigma_bias",
                             torch.tensor(sigma_bias_start, dtype=torch.float32))
        self.act = nn.GELU() if nonlinearity else nn.Identity()
        self.mix = nn.Linear(2 * r_enc, code_dim)
        self.in_channels = 2  # FLOPs counter compat

    def update_sigma_bias(self, epoch: int, total_epochs: int) -> None:
        progress = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
        val = self.sigma_bias_start * (1.0 - progress) + self.sigma_bias_end * progress
        self.current_sigma_bias.fill_(val)

    def _unit_norm_uv(self):
        # Per-rank complex unit norm: |u_r|^2 = sum(u_re^2 + u_im^2) = 1
        u_norm = (self.u_re.pow(2).sum(dim=-1, keepdim=True)
                  + self.u_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        v_norm = (self.v_re.pow(2).sum(dim=-1, keepdim=True)
                  + self.v_im.pow(2).sum(dim=-1, keepdim=True)
                  ).clamp(min=self.eps).sqrt()
        return (self.u_re / u_norm, self.u_im / u_norm,
                self.v_re / v_norm, self.v_im / v_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_re, u_im, v_re, v_im = self._unit_norm_uv()
        sigma = F.softplus(self.sigma_raw + self.current_sigma_bias)            # (R,)
        # Match ComplexLowRankEncoder math
        H_re = x[:, 0]
        H_im = x[:, 1]
        H_re_vre = torch.einsum("bij,rj->bri", H_re, v_re)
        H_re_vim = torch.einsum("bij,rj->bri", H_re, v_im)
        H_im_vre = torch.einsum("bij,rj->bri", H_im, v_re)
        H_im_vim = torch.einsum("bij,rj->bri", H_im, v_im)
        A = H_re_vre - H_im_vim
        B = H_re_vim + H_im_vre
        c_re = (torch.einsum("ri,bri->br", u_re, A)
                + torch.einsum("ri,bri->br", u_im, B))
        c_im = (torch.einsum("ri,bri->br", u_re, B)
                - torch.einsum("ri,bri->br", u_im, A))
        # Apply sigma per rank
        c_re = c_re * sigma                                                     # (B, R) * (R,)
        c_im = c_im * sigma
        c = torch.cat([c_re, c_im], dim=-1)
        c = self.act(c)
        return self.mix(c)


class SVDHybridRankDecoder2(nn.Module):
    """HybridRankDecoder where direct branch uses SVDLowRankDecoderSched."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 direct_ranks: int = 16, extra_ranks: int = 64,
                 extra_d_emb: int = 16,
                 gate_bias: float = -2.0, alpha_init: float = 1e-2,
                 sigma_bias_start: float = 2.0, sigma_bias_end: float = -2.0):
        super().__init__()
        self.direct = SVDLowRankDecoderSched(
            code_dim, h, w, ranks=direct_ranks,
            sigma_bias_start=sigma_bias_start,
            sigma_bias_end=sigma_bias_end,
        )
        self.extra = GatedFactoredResidualDecoder(
            code_dim, h, w,
            extra_ranks=extra_ranks, d_emb=extra_d_emb,
            gate_bias=gate_bias, alpha_init=alpha_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.direct(z) + self.extra(z)


class DCRNetV29SVD2(DCRNetV29):
    """v29 + SVD-encoder + SVD-decoder + sigma_bias schedule."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 r_enc: int = 512, nonlinearity: bool = True,
                 hybrid_direct_ranks: int = 32, hybrid_extra_ranks: int = 16,
                 hybrid_extra_d_emb: int = 16, hybrid_gate_bias: float = -2.0,
                 hybrid_alpha_init: float = 1e-2,
                 sigma_bias_start: float = 2.0, sigma_bias_end: float = -2.0,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size,
                         r_enc=r_enc, nonlinearity=nonlinearity,
                         hybrid_direct_ranks=hybrid_direct_ranks,
                         hybrid_extra_ranks=hybrid_extra_ranks,
                         hybrid_extra_d_emb=hybrid_extra_d_emb,
                         hybrid_gate_bias=hybrid_gate_bias,
                         hybrid_alpha_init=hybrid_alpha_init,
                         **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction

        # Replace encoder ComplexLowRankEncoder with SVD variant
        self.enc_neural = SVDComplexLowRankEncoder(
            h, w, code_dim, r_enc=r_enc, nonlinearity=nonlinearity,
            sigma_bias_start=sigma_bias_start,
            sigma_bias_end=sigma_bias_end,
        )
        # Replace decoder with SVD-direct + standard extra
        self.decoder = SVDHybridRankDecoder2(
            code_dim, h, w,
            direct_ranks=hybrid_direct_ranks,
            extra_ranks=hybrid_extra_ranks,
            extra_d_emb=hybrid_extra_d_emb,
            gate_bias=hybrid_gate_bias,
            alpha_init=hybrid_alpha_init,
            sigma_bias_start=sigma_bias_start,
            sigma_bias_end=sigma_bias_end,
        )

    def update_sigma_bias(self, epoch: int, total_epochs: int) -> None:
        # Forward to both encoder and decoder.direct
        if hasattr(self.enc_neural, "update_sigma_bias"):
            self.enc_neural.update_sigma_bias(epoch, total_epochs)
        if hasattr(self.decoder, "direct") and hasattr(self.decoder.direct, "update_sigma_bias"):
            self.decoder.direct.update_sigma_bias(epoch, total_epochs)


def dcrnet_v29svd2(
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
    sigma_bias_start: float = 2.0,
    sigma_bias_end: float = -2.0,
    **kwargs,
) -> DCRNetV29SVD2:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV29SVD2(
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
        sigma_bias_start=sigma_bias_start,
        sigma_bias_end=sigma_bias_end,
        **kwargs,
    )
