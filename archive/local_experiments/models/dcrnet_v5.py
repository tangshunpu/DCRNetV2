"""DCRNet v5: low-rank matched-filter encoder + low-rank decoder.

Same prior on both sides:
    H ≈ Σ_r (d_r + i e_r) ⊗ (a_r + i b_r)*

The decoder (v3) emits R rank-1 outer products. The encoder here is the
analytic *inverse*: a bank of learned (u_r, v_r) outer products, each
matched-filtering the input to produce one scalar coefficient

    c_{r,c} = u_r^T H_c v_r           c ∈ {re, im}

That is the most informative scalar you can extract from H under the
multipath prior — and it is just two GEMVs.

Encoder cost (R_enc=128, cr=4):
    - matched filtering: 2·R_enc·(H·W + W) MACs ≈ 270 K
    - mixer Linear(2·R_enc, code_dim) ≈ 131 K
    - total: ~400 K MACs vs v4's ~100 M (≈250x cheaper)

Encoder is *bilinear* in H — it sees only second-order structure. A
single GELU after matched filtering buys nonlinearity without changing
the FLOP picture (R_enc-dim activation is tiny).

Decoder is exactly v3's LowRankDecoder.
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = ["DCRNetV5", "LowRankEncoder", "LowRankDecoder", "dcrnet_v5"]


class LowRankDecoder(nn.Module):
    """Codeword -> R rank-1 complex outer products -> (2, H, W) tensor.

    Each rank contributes a complex-valued outer product
        H_r = (d_re + i d_im) ⊗ (a_re + i a_im)^*  (Hermitian outer)
    The total reconstruction is the sum over R, mapped to [0, 1].
    """

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
        u = self.proj(z).view(b, r, 2 * h + 2 * w)
        d_re, d_im, a_re, a_im = torch.split(u, [h, h, w, w], dim=-1)
        out_re = (torch.einsum("brh,brw->bhw", d_re, a_re)
                  + torch.einsum("brh,brw->bhw", d_im, a_im))
        out_im = (torch.einsum("brh,brw->bhw", d_im, a_re)
                  - torch.einsum("brh,brw->bhw", d_re, a_im))
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        out = torch.tanh(out)
        return (out + 1.0) * 0.5


class LowRankEncoder(nn.Module):
    """Bilinear matched-filter encoder.

    For each of R_enc learned outer products (u_r, v_r), compute one
    scalar per input channel:  c_{r,c} = u_r^T H_c v_r.

    Implemented as two batched GEMMs:  (H @ V) then (U^T · result).
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 r_enc: int = 128, nonlinearity: bool = True):
        super().__init__()
        self.r_enc = r_enc
        self.h, self.w = h, w
        # u_r ∈ R^h (delay axis), v_r ∈ R^w (angular axis), one per rank
        self.u = nn.Parameter(torch.empty(r_enc, h))
        self.v = nn.Parameter(torch.empty(r_enc, w))
        nn.init.trunc_normal_(self.u, std=h ** -0.5)
        nn.init.trunc_normal_(self.v, std=w ** -0.5)
        self.act = nn.GELU() if nonlinearity else nn.Identity()
        self.mix = nn.Linear(in_channels * r_enc, code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Compute c[b,c,r] = sum_{ij} u[r,i] x[b,c,i,j] v[r,j]
        # = einsum("ri,bcij,rj->bcr", u, x, v)
        Hv = torch.einsum("bcij,rj->bcir", x, self.v)        # (B, C, H, R)
        c = torch.einsum("ri,bcir->bcr", self.u, Hv)         # (B, C, R)
        c = c.flatten(1)                                      # (B, C*R)
        c = self.act(c)
        return self.mix(c)                                    # (B, code_dim)


class DCRNetV5(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        expansion: int = 1,        # kept for CLI symmetry; unused here
        r_enc: int = 128,
        ranks: int = 4,
        nonlinearity: bool = True,
        h: int = 32,
        w: int = 32,
        total_size: int | None = None,
        complex_encoder: bool = False,
    ):
        super().__init__()
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        del expansion

        if complex_encoder:
            assert in_channels == 2, "complex_encoder requires in_channels=2 (Re, Im)"
            from .dcrnet_v9 import ComplexLowRankEncoder
            self.encoder = ComplexLowRankEncoder(h, w, code_dim,
                                                 r_enc=r_enc, nonlinearity=nonlinearity)
        else:
            self.encoder = LowRankEncoder(in_channels, h, w, code_dim,
                                          r_enc=r_enc, nonlinearity=nonlinearity)
        self.decoder = LowRankDecoder(code_dim, h, w, ranks=ranks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2.0 - 1.0
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def dcrnet_v5(reduction: int = 4, expansion: int = 1, ranks: int = 8,
              r_enc: int = 128, h: int = 32, w: int = 32,
              total_size: int | None = None, complex_encoder: bool = False,
              **kwargs) -> DCRNetV5:
    return DCRNetV5(reduction=reduction, expansion=expansion, ranks=ranks,
                    r_enc=r_enc, h=h, w=w, total_size=total_size,
                    complex_encoder=complex_encoder, **kwargs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cr", type=int, default=4)
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--r-enc", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = dcrnet_v5(reduction=args.cr, ranks=args.ranks, r_enc=args.r_enc).to(device)
    x = torch.rand(2, 2, 32, 32, device=device)
    z = m.encode(x); y = m.decode(z)
    n_total = sum(p.numel() for p in m.parameters())
    n_enc = sum(p.numel() for p in m.encoder.parameters())
    n_dec = sum(p.numel() for p in m.decoder.parameters())
    print(f"cr={args.cr} ranks={args.ranks} r_enc={args.r_enc}  "
          f"codeword={tuple(z.shape)}  out={tuple(y.shape)}")
    print(f"total={n_total/1e3:.1f}K  encoder={n_enc/1e3:.1f}K  decoder={n_dec/1e3:.1f}K")
