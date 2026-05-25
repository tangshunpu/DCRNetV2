"""LRP: low-rank-prior CSI feedback model.

This is the public name for the v5 low-rank matched-filter model. LRP keeps
both encoder and decoder model-based and very small:

* Encoder: ``d`` learned bilinear matched filters ``u_r^T H v_r`` per CSI
  channel, followed by a lightweight linear mixer to the feedback codeword.
* Decoder: ``r`` learned complex rank-1 components generated from the codeword.

The paper/table configurations are exposed as factory functions:

* ``lrp_r4_d128``
* ``lrp_r4_d512``
* ``lrp_r8_d512``
* ``lrp_r16_d512``
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = [
    "LRP",
    "LRPEncoder",
    "LRPDecoder",
    "lrp",
    "lrp_r4_d128",
    "lrp_r4_d512",
    "lrp_r8_d512",
    "lrp_r16_d512",
]


class LRPDecoder(nn.Module):
    """Codeword -> ``r`` complex rank-1 outer products -> ``(B, 2, H, W)``."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32, ranks: int = 4):
        super().__init__()
        self.h, self.w, self.ranks = h, w, ranks
        per_rank = 2 * h + 2 * w
        self.proj = nn.Linear(code_dim, ranks * per_rank)
        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        h, w, r = self.h, self.w, self.ranks
        factors = self.proj(z).view(b, r, 2 * h + 2 * w)
        d_re, d_im, a_re, a_im = torch.split(factors, [h, h, w, w], dim=-1)
        out_re = (
            torch.einsum("brh,brw->bhw", d_re, a_re)
            + torch.einsum("brh,brw->bhw", d_im, a_im)
        )
        out_im = (
            torch.einsum("brh,brw->bhw", d_im, a_re)
            - torch.einsum("brh,brw->bhw", d_re, a_im)
        )
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        return (torch.tanh(out) + 1.0) * 0.5


class LRPEncoder(nn.Module):
    """Bilinear matched-filter encoder.

    For each of ``d`` learned outer products ``(u_k, v_k)``, compute one scalar
    per input channel: ``c_{k,c} = u_k^T H_c v_k``. The resulting ``C*d``
    features are mixed to the feedback codeword.
    """

    def __init__(
        self,
        in_channels: int,
        h: int,
        w: int,
        code_dim: int,
        d: int = 128,
        nonlinearity: bool = True,
    ):
        super().__init__()
        self.d = d
        self.h, self.w = h, w
        self.u = nn.Parameter(torch.empty(d, h))
        self.v = nn.Parameter(torch.empty(d, w))
        nn.init.trunc_normal_(self.u, std=h ** -0.5)
        nn.init.trunc_normal_(self.v, std=w ** -0.5)
        self.act = nn.GELU() if nonlinearity else nn.Identity()
        self.mix = nn.Linear(in_channels * d, code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). c[b,c,k] = sum_ij u[k,i] x[b,c,i,j] v[k,j]
        hv = torch.einsum("bcij,kj->bcik", x, self.v)
        c = torch.einsum("ki,bcik->bck", self.u, hv).flatten(1)
        return self.mix(self.act(c))


class LRP(nn.Module):
    """Low-rank-prior CSI feedback model.

    Args:
        reduction: Compression ratio denominator, e.g. ``4`` for eta=1/4.
        ranks: Decoder rank count ``r`` in the table.
        d: Encoder matched-filter count ``d`` in the table.
    """

    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        ranks: int = 4,
        d: int = 128,
        nonlinearity: bool = True,
        h: int = 32,
        w: int = 32,
        total_size: int | None = None,
    ):
        super().__init__()
        if total_size is None:
            total_size = in_channels * h * w
        code_dim = total_size // reduction
        self.reduction = reduction
        self.ranks = ranks
        self.d = d
        self.encoder = LRPEncoder(in_channels, h, w, code_dim, d=d, nonlinearity=nonlinearity)
        self.decoder = LRPDecoder(code_dim, h, w, ranks=ranks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x * 2.0 - 1.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def lrp(
    reduction: int = 4,
    ranks: int = 4,
    d: int = 128,
    **kwargs,
) -> LRP:
    """Build a configurable LRP model."""

    return LRP(reduction=reduction, ranks=ranks, d=d, **kwargs)


def lrp_r4_d128(reduction: int = 4, **kwargs) -> LRP:
    return lrp(reduction=reduction, ranks=4, d=128, **kwargs)


def lrp_r4_d512(reduction: int = 4, **kwargs) -> LRP:
    return lrp(reduction=reduction, ranks=4, d=512, **kwargs)


def lrp_r8_d512(reduction: int = 4, **kwargs) -> LRP:
    return lrp(reduction=reduction, ranks=8, d=512, **kwargs)


def lrp_r16_d512(reduction: int = 4, **kwargs) -> LRP:
    return lrp(reduction=reduction, ranks=16, d=512, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LRP sanity check")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--d", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lrp(reduction=args.cr, ranks=args.ranks, d=args.d).to(device)
    x = torch.rand(2, 2, 32, 32, device=device)
    z = model.encode(x)
    y = model.decode(z)
    params = sum(p.numel() for p in model.parameters())
    print(f"cr={args.cr} r={args.ranks} d={args.d} codeword={tuple(z.shape)} out={tuple(y.shape)}")
    print(f"params={params/1e3:.1f}K")
