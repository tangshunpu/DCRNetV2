"""DCRNetV2 — self-contained model file for the public production variants.

Variants (factory functions):
    dcrnetv2_mini(reduction)       3.01 M cr=4   r_enc=128, hybrid 10+8
    dcrnetv2_small(reduction)      4.07 M cr=4   r_enc=256, ranks=16
    dcrnetv2_base(reduction)       5.41 M cr=4   r_enc=512, ranks=16
    dcrnetv2_unified(reduction)    6.77 M cr=4   r_enc=512, hybrid 32+16
    dcrnetv2_large_out(reduction)  7.00 M cr=4   r_enc=512, hybrid 32+32, α=0.05, gate=-2
    dcrnetv2_large_in4(reduction)  7.00 M cr=4   r_enc=512, hybrid 32+32, α=0.05, gate=-4
                                    (indoor cr=4 SOTA; for cr=16 indoor use FT-recipe)

All variants share:
  Encoder = ComplexLowRankEncoder (bilinear Hermitian matched-filter)
          + DilateEncoderPath (DCREncoderBlock + Linear)
          + EncoderFuseMLP (concat + 2-layer MLP residual)
  Decoder = LowRankDecoder | HybridRankDecoder (direct + cheap z-conditioned extras)
  Refine  = IterativeRefine (DCRDecoderRefine stack)

Input/output: (B, 2, 32, 32) CSI in [0, 1], compressed via codeword of length
2048/reduction.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


__all__ = [
    "DCRNetV2",
    "dcrnetv2_mini",
    "dcrnetv2_small",
    "dcrnetv2_base",
    "dcrnetv2_unified",
    "dcrnetv2_large_out",
    "dcrnetv2_large_in4",
]


# ============================================================================
# Building blocks
# ============================================================================


def _equ_conv(kernel_size: int, dilation: int) -> int:
    return kernel_size + (kernel_size - 1) * (dilation - 1) if dilation > 1 else kernel_size


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, dilation=1):
        if not isinstance(kernel_size, int):
            padding = [(_equ_conv(i, dilation) - 1) // 2 for i in kernel_size]
        else:
            padding = (_equ_conv(kernel_size, dilation) - 1) // 2
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, dilation=dilation, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.groups, c // self.groups, h, w).transpose(1, 2).contiguous().view(b, -1, h, w)


class DCREncoderBlock(nn.Module):
    """DCRNet v1 encoder block: parallel asymmetric dilated convs + 1x1 fuse."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBN(2, 2, [3, 1], dilation=1), nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1, 3], dilation=1), nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [3, 1], dilation=2), nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1, 3], dilation=2), nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [3, 1], dilation=3), nn.PReLU(num_parameters=2, init=0.3),
            ConvBN(2, 2, [1, 3], dilation=3),
        )
        self.conv2 = ConvBN(2, 2, 3, dilation=1)
        self.prelu1 = nn.PReLU(num_parameters=4, init=0.3)
        self.prelu2 = nn.PReLU(num_parameters=2, init=0.3)
        self.conv1x1 = ConvBN(4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.prelu1(torch.cat((self.conv1(x), self.conv2(x)), dim=1))
        return self.prelu2(x + self.conv1x1(res))


class DCRDecoderRefine(nn.Module):
    """Residual refine block: two parallel grouped+shuffled dilated paths, 1x1 fuse.

    Gate-init 1e-3 keeps the block near-identity at step 0.
    """

    def __init__(self, width: int = 2):
        super().__init__()
        assert width >= 2 and width % 2 == 0
        groups = max(1, width // 2)
        self.path1 = nn.Sequential(OrderedDict([
            ("conv_3x3", ConvBN(2, width, 3, dilation=2)),
            ("prelu1", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x5", ConvBN(width, width, [3, 1], groups=groups, dilation=3)),
            ("shuffle1", ChannelShuffle(groups)),
            ("prelu2", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_5x1", ConvBN(width, width, [1, 3], groups=groups, dilation=3)),
            ("shuffle2", ChannelShuffle(groups)),
            ("prelu4", nn.PReLU(num_parameters=width, init=0.3)),
            ("conv_1x1", ConvBN(width, 2, 3)),
        ]))
        self.path2 = nn.Sequential(
            ConvBN(2, width, [1, 3]),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [5, 1], groups=groups),
            ChannelShuffle(groups),
            nn.PReLU(num_parameters=width, init=0.3),
            ConvBN(width, width, [1, 5], groups=groups),
            ChannelShuffle(groups),
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


class IterativeRefine(nn.Module):
    """K stacked DCRDecoderRefine blocks (residual, near-identity at init)."""

    def __init__(self, width: int = 2, K: int = 1):
        super().__init__()
        assert K >= 1
        self.stages = nn.ModuleList([DCRDecoderRefine(width=width) for _ in range(K)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s(x)
        return x


# ============================================================================
# Encoders
# ============================================================================


class ComplexLowRankEncoder(nn.Module):
    """Complex Hermitian bilinear matched-filter encoder.

    c_r = u_r^H · H · v_r  with all 4 cross-terms preserved
    (u, v have learnable real and imaginary parts; H = H_re + i H_im).
    """

    def __init__(self, h: int, w: int, code_dim: int, r_enc: int = 512,
                 nonlinearity: bool = True):
        super().__init__()
        self.r_enc, self.h, self.w = r_enc, h, w
        self.u_re = nn.Parameter(torch.empty(r_enc, h))
        self.u_im = nn.Parameter(torch.empty(r_enc, h))
        self.v_re = nn.Parameter(torch.empty(r_enc, w))
        self.v_im = nn.Parameter(torch.empty(r_enc, w))
        nn.init.trunc_normal_(self.u_re, std=h ** -0.5)
        nn.init.trunc_normal_(self.u_im, std=h ** -0.5)
        nn.init.trunc_normal_(self.v_re, std=w ** -0.5)
        nn.init.trunc_normal_(self.v_im, std=w ** -0.5)
        self.act = nn.GELU() if nonlinearity else nn.Identity()
        self.mix = nn.Linear(2 * r_enc, code_dim)
        self.in_channels = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_re, H_im = x[:, 0], x[:, 1]
        A = (torch.einsum("bij,rj->bri", H_re, self.v_re)
             - torch.einsum("bij,rj->bri", H_im, self.v_im))
        B = (torch.einsum("bij,rj->bri", H_re, self.v_im)
             + torch.einsum("bij,rj->bri", H_im, self.v_re))
        c_re = (torch.einsum("ri,bri->br", self.u_re, A)
                + torch.einsum("ri,bri->br", self.u_im, B))
        c_im = (torch.einsum("ri,bri->br", self.u_re, B)
                - torch.einsum("ri,bri->br", self.u_im, A))
        return self.mix(self.act(torch.cat([c_re, c_im], dim=-1)))


class DilateEncoderPath(nn.Module):
    """DCREncoderBlock + Linear(2HW, code_dim) with LN + learnable gate."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int):
        super().__init__()
        self.block = DCREncoderBlock()
        self.proj = nn.Linear(in_channels * h * w, code_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.block(x).flatten(1))
        return self.gate * self.norm(z)


class EncoderFuseMLP(nn.Module):
    """Concat + 2-layer MLP residual fusion: output = z_a + z_b + MLP(concat)."""

    def __init__(self, code_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(2 * code_dim, hidden)
        self.fc2 = nn.Linear(hidden, code_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return z_a + z_b + self.fc2(torch.nn.functional.gelu(self.fc1(torch.cat([z_a, z_b], dim=-1))))


# ============================================================================
# Decoders
# ============================================================================


class LowRankDecoder(nn.Module):
    """Codeword -> R rank-1 complex outer products -> (B, 2, H, W) in [0, 1]."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32, ranks: int = 16):
        super().__init__()
        self.h, self.w, self.ranks = h, w, ranks
        self.proj = nn.Linear(code_dim, ranks * (2 * h + 2 * w))
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
        return (torch.tanh(out) + 1.0) * 0.5


class GatedFactoredResidualDecoder(nn.Module):
    """Cheap z-conditioned residual rank bank.

    Each extra rank generates a (d_r(z), a_r(z)) complex rank-1 component
    factored through a d_emb-dim bottleneck. Per-rank sigmoid gate controls
    contribution, global alpha scales the residual.
    """

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 extra_ranks: int = 16, d_emb: int = 16,
                 gate_bias: float = -2.0, alpha_init: float = 1e-2):
        super().__init__()
        self.h, self.w = h, w
        self.extra_ranks, self.d_emb = extra_ranks, d_emb

        self.code_proj = nn.Linear(code_dim, extra_ranks * d_emb)
        nn.init.trunc_normal_(self.code_proj.weight, std=0.02)
        nn.init.zeros_(self.code_proj.bias)

        self.W_d_re = nn.Parameter(torch.empty(extra_ranks, d_emb, h))
        self.W_d_im = nn.Parameter(torch.empty(extra_ranks, d_emb, h))
        self.W_a_re = nn.Parameter(torch.empty(extra_ranks, d_emb, w))
        self.W_a_im = nn.Parameter(torch.empty(extra_ranks, d_emb, w))
        for W in (self.W_d_re, self.W_d_im, self.W_a_re, self.W_a_im):
            nn.init.trunc_normal_(W, std=d_emb ** -0.5)

        self.rank_gate = nn.Linear(code_dim, extra_ranks)
        nn.init.zeros_(self.rank_gate.weight)
        nn.init.constant_(self.rank_gate.bias, gate_bias)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        F = self.code_proj(z).view(B, self.extra_ranks, self.d_emb)
        d_re = torch.einsum("brd,rdh->brh", F, self.W_d_re)
        d_im = torch.einsum("brd,rdh->brh", F, self.W_d_im)
        a_re = torch.einsum("brd,rdw->brw", F, self.W_a_re)
        a_im = torch.einsum("brd,rdw->brw", F, self.W_a_im)
        out_re = (torch.einsum("brh,brw->brhw", d_re, a_re)
                  + torch.einsum("brh,brw->brhw", d_im, a_im))
        out_im = (torch.einsum("brh,brw->brhw", d_im, a_re)
                  - torch.einsum("brh,brw->brhw", d_re, a_im))
        rank_out = torch.stack([out_re, out_im], dim=2)
        gate = torch.sigmoid(self.rank_gate(z)).view(B, self.extra_ranks, 1, 1, 1)
        return self.alpha * (gate * rank_out).sum(dim=1)


class HybridRankDecoder(nn.Module):
    """Direct LowRankDecoder + GatedFactoredResidualDecoder (z-conditioned extras)."""

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 direct_ranks: int = 32, extra_ranks: int = 16,
                 extra_d_emb: int = 16,
                 gate_bias: float = -2.0, alpha_init: float = 1e-2):
        super().__init__()
        self.direct = LowRankDecoder(code_dim, h, w, ranks=direct_ranks)
        self.extra = GatedFactoredResidualDecoder(
            code_dim, h, w,
            extra_ranks=extra_ranks, d_emb=extra_d_emb,
            gate_bias=gate_bias, alpha_init=alpha_init,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.direct(z) + self.extra(z)


# ============================================================================
# Main model
# ============================================================================


class DCRNetV2(nn.Module):
    """DCRNetV2 main model used by all public factory functions."""

    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        # encoder
        r_enc: int = 512,
        enc_fuse_mlp: int = 64,
        # decoder
        ranks: int = 16,                  # used when hybrid_decoder=False
        hybrid_decoder: bool = False,
        hybrid_direct_ranks: int = 32,
        hybrid_extra_ranks: int = 16,
        hybrid_extra_d_emb: int = 16,
        hybrid_gate_bias: float = -2.0,
        hybrid_alpha_init: float = 1e-2,
        # refine
        refine_width: int = 2,
        refine_K: int = 1,
    ):
        super().__init__()
        h, w = 32, 32
        code_dim = 2048 // reduction

        # Encoder (always: bilinear + dilate + concat-MLP fusion)
        self.enc_neural = ComplexLowRankEncoder(h, w, code_dim, r_enc=r_enc)
        self.enc_dilate = DilateEncoderPath(in_channels, h, w, code_dim)
        self.fuse_enc = EncoderFuseMLP(code_dim, hidden=enc_fuse_mlp)

        # Decoder
        if hybrid_decoder:
            self.decoder = HybridRankDecoder(
                code_dim, h, w,
                direct_ranks=hybrid_direct_ranks,
                extra_ranks=hybrid_extra_ranks,
                extra_d_emb=hybrid_extra_d_emb,
                gate_bias=hybrid_gate_bias,
                alpha_init=hybrid_alpha_init,
            )
        else:
            self.decoder = LowRankDecoder(code_dim, h, w, ranks=ranks)

        self.refine = IterativeRefine(width=refine_width, K=refine_K)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_neural(x_centered)
        z_b = self.enc_dilate(x)
        return self.fuse_enc(z_a, z_b)

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        out = self.refine(self.decoder(z))
        return out.clamp(0.0, 1.0) if clamp else out

    def forward(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return self.decode(self.encode(x), clamp=clamp)


# ============================================================================
# Factory functions for the public production variants
# ============================================================================


def dcrnetv2_mini(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-mini: 3.01M FLOPs cr=4. r_enc=128 + hybrid direct=10 + extra=8 d=8."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=128,
        enc_fuse_mlp=32,
        hybrid_decoder=True,
        hybrid_direct_ranks=10,
        hybrid_extra_ranks=8,
        hybrid_extra_d_emb=8,
        hybrid_gate_bias=-2.0,
        hybrid_alpha_init=1e-2,
    )


def dcrnetv2_small(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-small: 4.07M FLOPs cr=4. r_enc=256, ranks=16."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=256,
        enc_fuse_mlp=64,
        ranks=16,
        hybrid_decoder=False,
    )


def dcrnetv2_base(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-base: 5.41M FLOPs cr=4. r_enc=512, ranks=16."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=512,
        enc_fuse_mlp=64,
        ranks=16,
        hybrid_decoder=False,
    )


def dcrnetv2_unified(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-unified: 6.77M FLOPs cr=4. r_enc=512 + hybrid direct=32 + extra=16."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=512,
        enc_fuse_mlp=64,
        hybrid_decoder=True,
        hybrid_direct_ranks=32,
        hybrid_extra_ranks=16,
        hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0,
        hybrid_alpha_init=1e-2,
    )


def dcrnetv2_large_out(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-large-out: outdoor SOTA. α=0.05, gate=-2."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=512,
        enc_fuse_mlp=64,
        hybrid_decoder=True,
        hybrid_direct_ranks=32,
        hybrid_extra_ranks=32,
        hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0,
        hybrid_alpha_init=0.05,
    )


def dcrnetv2_large_in4(reduction: int = 4) -> DCRNetV2:
    """DCRNetV2-large-in4: indoor cr=4 SOTA. α=0.05, gate=-4 (extras closed init)."""
    return DCRNetV2(
        reduction=reduction,
        r_enc=512,
        enc_fuse_mlp=64,
        hybrid_decoder=True,
        hybrid_direct_ranks=32,
        hybrid_extra_ranks=32,
        hybrid_extra_d_emb=16,
        hybrid_gate_bias=-4.0,
        hybrid_alpha_init=0.05,
    )


if __name__ == "__main__":
    # Quick sanity check + FLOPs
    import thop
    for name, fn in [
        ("mini", dcrnetv2_mini),
        ("small", dcrnetv2_small),
        ("base", dcrnetv2_base),
        ("unified", dcrnetv2_unified),
        ("large-out", dcrnetv2_large_out),
        ("large-in4", dcrnetv2_large_in4),
    ]:
        for cr in [4, 8, 16, 32]:
            m = fn(reduction=cr)
            x = torch.rand(1, 2, 32, 32)
            f, _ = thop.profile(m, inputs=(x,), verbose=False)
            n = sum(p.numel() for p in m.parameters())
            print(f"{name:10s} cr={cr:2d}  thop={f/1e6:.3f}M  params={n/1e3:.1f}K")
