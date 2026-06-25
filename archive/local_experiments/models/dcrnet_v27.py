"""DCRNet v27: v9u dilate path + attention residual injected between conv and Linear.

v26 failure mode: replacing v9u's Linear(2048, code_dim) with PatchEmbed +
mean-pool + Linear(d_model, code_dim) lost too much dense compression
capacity. Linear(2048, code_dim) is the *workhorse* compressor and must
stay.

v27 keeps v9u's full DilateEncoderPath structure, inserts a lightweight
attention RESIDUAL between DCREncoderBlock and the Linear projection.

ENCODER (2 branches):
    z_a = ComplexLowRankEncoder(x)                     [main: low-rank]
    z_b = DilateAttnEncoderPath(x)                     [supplement]
        DCREncoderBlock                                [v9u 7-conv dilated]
        + AttnResidual (small gate, identity init)     [global pair refine]
        Linear(2*H*W, code_dim) + LN + gate            [v9u dense compress]

AttnResidual:
    PatchEmbed 2 -> d_model=16, 8x8 patches (16 tokens)
    1-layer Transformer (MHA + small FFN)
    Transposed-Conv 16 -> 2 (back to 32x32 spatial)
    residual gate * to add to conv output

DECODER: HybridRankDecoder + plain IterativeRefine + hsigmoid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dcrnet import ConvBN
from .dcrnet_v8 import DilateEncoderPath
from .dcrnet_v9 import DCRNetV9


__all__ = ["DCRNetV27", "AttnResidual", "DilateAttnEncoderPath", "dcrnet_v27"]


class _TransformerBlock(nn.Module):
    """Pre-LN MHA + FFN."""
    def __init__(self, d_model, n_heads, ffn_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, t):
        n = self.norm1(t)
        a, _ = self.attn(n, n, n, need_weights=False)
        t = t + a
        t = t + self.ffn(self.norm2(t))
        return t


class AttnResidual(nn.Module):
    """Patchify -> 1 transformer block -> token-Linear + bilinear upsample,
    applied as residual.

    Avoids the expensive ConvTranspose2d(k=patch) (~2M MACs for k=8). Each
    token is projected to a single (channels,) value via Linear, placed
    in a low-res grid, then bilinearly upsampled to spatial resolution.

    Identity-init: token_proj.weight=0 -> residual contributes ~0 at step 0.
    Cost on 32x32 / patch=8 / d=16: ~80K MACs.
    """
    def __init__(self, channels: int = 2, h: int = 32, w: int = 32,
                 patch_size: int = 8, d_model: int = 16, n_heads: int = 4,
                 ffn_hidden: int = 32):
        super().__init__()
        assert h % patch_size == 0 and w % patch_size == 0
        self.h, self.w, self.ps = h, w, patch_size
        self.grid_h = h // patch_size
        self.grid_w = w // patch_size
        self.patch = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.block = _TransformerBlock(d_model, n_heads, ffn_hidden)
        # Token -> 1 (channels,)-valued sample per token; upsample bilinearly.
        self.token_proj = nn.Linear(d_model, channels)
        nn.init.zeros_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)        # (B, n_tok, d)
        t = self.block(t)
        s = self.token_proj(t)                              # (B, n_tok, channels)
        s = s.transpose(1, 2).reshape(
            x.size(0), -1, self.grid_h, self.grid_w
        )                                                    # (B, channels, gh, gw)
        y = F.interpolate(s, size=(self.h, self.w),
                          mode='bilinear', align_corners=False)
        return x + self.gate * y


class DilateAttnEncoderPath(DilateEncoderPath):
    """v9u's DilateEncoderPath with an AttnResidual inserted between
    DCREncoderBlock and the Linear projection."""

    def __init__(self, in_channels: int, h: int, w: int, code_dim: int,
                 attn_patch_size: int = 8, attn_d_model: int = 16,
                 attn_n_heads: int = 4, attn_ffn_hidden: int = 32):
        super().__init__(in_channels, h, w, code_dim)
        # Insert attention residual on the 2-channel feature map.
        self.attn = AttnResidual(
            channels=in_channels, h=h, w=w,
            patch_size=attn_patch_size, d_model=attn_d_model,
            n_heads=attn_n_heads, ffn_hidden=attn_ffn_hidden,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)                                    # v9u 7-conv dilated
        if self._use_sa_enc:
            mask_in = torch.cat([h.max(1, keepdim=True)[0], h.mean(1, keepdim=True)], dim=1)
            h = h * torch.sigmoid(self.sa_enc(mask_in))
        h = self.attn(h)                                     # attention residual
        z = self.proj(h.flatten(1))
        z = self.norm(z)
        return self.gate * z


class DCRNetV27(DCRNetV9):
    """2-branch: ComplexLowRankEncoder + DilateAttnEncoderPath (dilated + attn residual)."""

    def __init__(self, *args, h: int = 32, w: int = 32,
                 in_channels: int = 2, reduction: int = 4,
                 total_size: int | None = None,
                 attn_patch_size: int = 8, attn_d_model: int = 16,
                 attn_n_heads: int = 4, attn_ffn_hidden: int = 32,
                 **kwargs):
        super().__init__(*args, h=h, w=w, in_channels=in_channels,
                         reduction=reduction, total_size=total_size, **kwargs)
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        if self.use_dilate_path:
            self.enc_dilate = DilateAttnEncoderPath(
                in_channels, h, w, code_dim,
                attn_patch_size=attn_patch_size, attn_d_model=attn_d_model,
                attn_n_heads=attn_n_heads, attn_ffn_hidden=attn_ffn_hidden,
            )

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            return F.relu6(refined * 6.0) / 6.0
        return refined


def dcrnet_v27(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    refine_width: int = 2,
    refine_K: int = 1,
    attn_d_model: int = 16,
    attn_n_heads: int = 4,
    attn_patch_size: int = 8,
    attn_ffn_hidden: int = 32,
    **kwargs,
) -> DCRNetV27:
    for k in ("use_film", "complex_encoder", "hybrid_decoder",
              "hybrid_direct_ranks", "hybrid_extra_ranks", "hybrid_extra_d_emb",
              "hybrid_gate_bias", "hybrid_alpha_init", "enc_fuse_mlp",
              "use_dilate_path"):
        kwargs.pop(k, None)
    return DCRNetV27(
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
        attn_d_model=attn_d_model, attn_n_heads=attn_n_heads,
        attn_patch_size=attn_patch_size, attn_ffn_hidden=attn_ffn_hidden,
        **kwargs,
    )
