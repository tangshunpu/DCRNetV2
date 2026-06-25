"""DCRNet v8: parallel low-rank + dilated-conv encoder, low-rank decoder + DCR refine.

Encoder (two parallel paths, codewords summed):
  - path A — bilinear matched filter (v5 ``LowRankEncoder``):
        z_A = mix( {u_r^T H_c v_r}_{r,c} )
    Captures the rank-1 multipath structure that dominates COST2100 CSI.
  - path B — v1 ``DCREncoderBlock`` + flatten + Linear(2048, code_dim):
        z_B = W · vec(DCREncoderBlock(H))
    Captures the asymmetric dilated structure (sidelobes, off-grid
    multipaths) that pure outer products cannot extract.
  - z = z_A + z_B

Path B's Linear is zero-init, so the model starts identical to v5 and
gradually learns to allocate codeword bits to path B where the rank-1
prior leaks.

Decoder (serial, unchanged from earlier v8 draft):
  - v3/v5 ``LowRankDecoder`` emits a sum of R rank-1 complex outer products.
  - v1-style ``DCRDecoderRefine`` (width-parameterized DCRDecoderBlock)
    post-refines as a residual; zero-init fusing 1x1 → starts at exact
    identity.

FLOP budget at cr=16 (thop MACs):
  enc path A LowRank Linear  0.033 M
  enc path B DCREncoderBlock 0.201 M
  enc path B Linear(2048,128) 0.262 M
  dec LowRank Linear         0.066 M
  dec refine (width=2)       0.264 M
  ------------------------------
  total                      0.826 M  <  1 M
``refine_dec_width=4`` lifts dec refine to 0.489 M and total to ~1.05 M
(over budget at cr=16 but fine at cr=4/8 where the constraint isn't
binding); default is therefore width=2.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .dcrnet import ConvBN, Channel_shuffle, DCREncoderBlock
from .dcrnet_v5 import LowRankEncoder, LowRankDecoder


__all__ = ["DCRNetV8", "dcrnet_v8"]


class DCRDecoderRefine(nn.Module):
    """v1 ``DCRDecoderBlock`` with width parameterized directly.

    Two parallel paths of asymmetric grouped dilated convs fused by a
    1x1, with a residual skip on the (2, H, W) input. Identical to
    v1's block at width=8 (== ``expansion=1``); smaller widths shrink
    both the channel count and the group count (groups = width // 2)
    so the structure is preserved.

    Refine output: ``out = prelu2(x + gate * conv1x1(...))``. The
    learnable scalar ``gate`` is zero-initialized so the block starts
    at exact identity (output == input) while *all internal parameters
    still receive gradient* (vs zero-init on conv1x1 weight, which
    would dead-end the gradient chain at step 0).
    """

    def __init__(self, width: int = 2):
        super().__init__()
        assert width >= 2 and width % 2 == 0, "width must be an even int >= 2"
        groups = max(1, width // 2)
        self.path1 = nn.Sequential(OrderedDict([
            ("conv_3x3", ConvBN(2, width, 3, dilation=2)),
            ("prelu1", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x5", ConvBN(width, width, [3, 1], groups=groups, dilation=3)),
            ("shuffle1", Channel_shuffle(groups)),
            ("prelu2", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_5x1", ConvBN(width, width, [1, 3], groups=groups, dilation=3)),
            ("shuffle2", Channel_shuffle(groups)),
            ("prelu4", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x1", ConvBN(width, 2, 3)),
        ]))
        self.path2 = nn.Sequential(
            ConvBN(2, width, [1, 3]),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [5, 1], groups=groups),
            Channel_shuffle(groups),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [1, 5], groups=groups),
            Channel_shuffle(groups),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, 2, [3, 1]),
        )
        self.prelu = nn.PReLU(num_parameters=4, init=0.3)
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv1x1 = ConvBN(4, 2, 1)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prelu(torch.cat((self.path1(x), self.path2(x)), dim=1))
        return self.prelu2(x + self.gate * self.conv1x1(out))


class DilateEncoderPath(nn.Module):
    """v1 DCREncoderBlock + Flatten + Linear projection to codeword.

    Parallel partner to LowRankEncoder. Output goes through a (no-affine)
    LayerNorm so its magnitude is bounded irrespective of the conv
    block's activations, then is scaled by a learnable scalar ``gate``
    (init 0) before being summed into the codeword.

    Without the LN+gate, the projection's weight grows freely once the
    bias starts updating, ``||z_b||`` blows up to 1000x ``||z_a||``,
    and the parallel sum is dominated by noise — the symptom we
    observed in the first v8 training.
    """

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int):
        super().__init__()
        self.block = DCREncoderBlock()
        self.proj = nn.Linear(in_channels * h * w, code_dim)
        # Standard small-std init lets the projection train from step 1.
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        # No learnable affine — pure magnitude normalization.
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))
        # Optional SpatialGate on the conv-block output (before flatten/proj):
        # locates angular-delay spikes before they get projected into the codeword.
        # DCRNET_V9_SA_ENC=1 enables. Zero conv weight + bias=3 → init mask ≈ 1.
        import os as _os
        self._use_sa_enc = _os.environ.get('DCRNET_V9_SA_ENC', '0') == '1'
        if self._use_sa_enc:
            self.sa_enc = nn.Conv2d(2, 1, 7, padding=3, bias=True)
            nn.init.zeros_(self.sa_enc.weight)
            nn.init.constant_(self.sa_enc.bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # v1 block was trained on [0,1] CSI — feed raw input.
        h = self.block(x)
        if self._use_sa_enc:
            import torch as _t
            mask_in = _t.cat([h.max(1, keepdim=True)[0], h.mean(1, keepdim=True)], dim=1)
            h = h * _t.sigmoid(self.sa_enc(mask_in))
        z = self.proj(h.flatten(1))
        z = self.norm(z)
        return self.gate * z


class DCRNetV8(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        expansion: int = 1,
        r_enc: int = 128,
        ranks: int = 4,
        refine_dec_width: int | None = None,
        nonlinearity: bool = True,
    ):
        super().__init__()
        total_size, h, w = 2048, 32, 32
        code_dim = total_size // reduction
        # If refine_dec_width is not explicitly given, scale with expansion:
        # exp=1 -> width=2 (matches the cr=16 < 1 M budget), exp=4 -> width=8,
        # mirroring v1's "expansion makes the refine wider" semantics.
        if refine_dec_width is None:
            refine_dec_width = 2 * expansion

        self.enc_lowrank = LowRankEncoder(in_channels, h, w, code_dim,
                                          r_enc=r_enc, nonlinearity=nonlinearity)
        self.enc_dilate = DilateEncoderPath(in_channels, h, w, code_dim)
        self.decoder = LowRankDecoder(code_dim, h, w, ranks=ranks)
        self.dec_refine = DCRDecoderRefine(width=refine_dec_width)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # path A: matched filter sees recentred input
        x_lr = x * 2.0 - 1.0
        z_a = self.enc_lowrank(x_lr)
        # path B: v1 block sees raw [0,1] CSI
        z_b = self.enc_dilate(x)
        return z_a + z_b

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        coarse = self.decoder(z)             # (B, 2, H, W) in [0, 1]
        refined = self.dec_refine(coarse)    # residual: refined ≈ coarse at init
        return refined.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def dcrnet_v8(reduction: int = 4, expansion: int = 1, ranks: int = 4,
              r_enc: int = 128, refine_dec_width: int | None = None, **kwargs) -> DCRNetV8:
    return DCRNetV8(reduction=reduction, expansion=expansion, ranks=ranks,
                    r_enc=r_enc, refine_dec_width=refine_dec_width, **kwargs)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cr", type=int, default=16)
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--r-enc", type=int, default=128)
    p.add_argument("--refine-dec-width", type=int, default=2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = dcrnet_v8(reduction=args.cr, ranks=args.ranks, r_enc=args.r_enc,
                  refine_dec_width=args.refine_dec_width).to(device)
    x = torch.rand(2, 2, 32, 32, device=device)
    z = m.encode(x); y = m.decode(z)
    n_total = sum(p.numel() for p in m.parameters())
    n_lr = sum(p.numel() for p in m.enc_lowrank.parameters())
    n_di = sum(p.numel() for p in m.enc_dilate.parameters())
    n_dec = sum(p.numel() for p in m.decoder.parameters())
    n_ref = sum(p.numel() for p in m.dec_refine.parameters())
    print(f"cr={args.cr} ranks={args.ranks} r_enc={args.r_enc} "
          f"refine_dec_width={args.refine_dec_width}  "
          f"codeword={tuple(z.shape)}  out={tuple(y.shape)}")
    print(f"total={n_total/1e3:.1f}K  enc_lowrank={n_lr/1e3:.1f}K  "
          f"enc_dilate={n_di/1e3:.1f}K  dec={n_dec/1e3:.1f}K  "
          f"dec_refine={n_ref/1e3:.2f}K")
