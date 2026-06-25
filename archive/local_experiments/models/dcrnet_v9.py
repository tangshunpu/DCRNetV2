"""DCRNet v9: input-adaptive neural low-rank encoder + iterative DCR refinement.

Core idea — keep v5's bilinear "matched-filter" measurement
    c_{r,c} = u_r^T H_c v_r
but generate (u_r, v_r) with a tiny hyper-network that is **per-sample
content adaptive**:

    e_r       : learnable rank embedding (R, D)
    γ(x), β(x): FiLM produced by a small context CNN over H
    e_r'      = γ(x) ⊙ e_r + β(x)
    u_r       = e_r' @ W_U   (D -> H)
    v_r       = e_r' @ W_V   (D -> W)

This is the "neural-network low-rank" requested by the spec: rank-1
factors are produced by a NN rather than being free parameters. At
init γ = 1, β = 0 (film bias zero-init) so the encoder reduces to a
static low-rank bank — identical in spirit to v5 — and adaptation is
learned gradually.

Decoder = v5 ``LowRankDecoder`` (rank-R complex outer product) followed
by K stacked v8 ``DCRDecoderRefine`` blocks (iterative residual refine).
Each refine stage is residual with a learnable scalar gate initialized
to ~0, so the model starts at the rank-R reconstruction and only learns
to refine on top.

FLOP budget (cr=4, defaults r_enc=512, ranks=16, d_emb=64, refine_K=2,
refine_width=8, dilate_path=True):
    enc neural (FiLM + W_U/W_V + bilinear + mix)   ~3.8 M
    enc dilate (DCREncoderBlock + 2048->512 lin)   ~1.3 M
    decoder lowrank (512->2048 lin + outer)        ~1.1 M
    refine (2x DCRDecoderRefine width=8)           ~1.4 M
    ------------------------------------------------------
    total                                          ~7.6 M  (cr=4)
At cr=16/32 the budget drops to ~5-6M (Linear projection cost shrinks
with code_dim). All configs fit comfortably under 20 M.

The optional ``dilate_path=False`` flag and the (r_enc, ranks,
refine_width, refine_K) knobs let us sweep capacity within the budget.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .dcrnet import ConvBN
from .dcrnet_v5 import LowRankDecoder, LowRankEncoder
from .dcrnet_v8 import DCRDecoderRefine, DilateEncoderPath


__all__ = ["DCRNetV9", "NeuralLowRankEncoder", "NeuralLowRankDecoder",
           "ComplexLowRankEncoder", "IterativeRefine", "RankTokenAttention",
           "GatedFactoredResidualDecoder", "HybridRankDecoder",
           "dcrnet_v9"]


class ComplexLowRankEncoder(nn.Module):
    """True complex Hermitian bilinear: c_r = u_r^H · H · v_r ∈ ℂ.

    Treats x[:, 0] = H_re, x[:, 1] = H_im. Learnable parameters
    u_re, u_im ∈ R^{R,H} and v_re, v_im ∈ R^{R,W} encode complex u,v.

    Math expansion (u = u_re + j·u_im, v = v_re + j·v_im, H = H_re + j·H_im):
        let A = H_re v_re - H_im v_im   (real part of H·v)
        let B = H_re v_im + H_im v_re   (imag part of H·v)
        u^H (A + jB) = (u_re^T A + u_im^T B) + j (u_re^T B - u_im^T A)

    Output: 2R real values (Re, Im of each c_r) → Linear(2R, code_dim).
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
        # in_channels stored for FLOPs counter compatibility
        self.in_channels = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, H, W) where ch0 = H_re, ch1 = H_im
        H_re = x[:, 0]
        H_im = x[:, 1]
        # H · v (complex): A + jB where A, B are real (B, R, H) tensors
        H_re_vre = torch.einsum("bij,rj->bri", H_re, self.v_re)
        H_re_vim = torch.einsum("bij,rj->bri", H_re, self.v_im)
        H_im_vre = torch.einsum("bij,rj->bri", H_im, self.v_re)
        H_im_vim = torch.einsum("bij,rj->bri", H_im, self.v_im)
        A = H_re_vre - H_im_vim
        B = H_re_vim + H_im_vre
        # u^H · (A + jB) = (u_re^T A + u_im^T B) + j (u_re^T B - u_im^T A)
        c_re = (torch.einsum("ri,bri->br", self.u_re, A)
                + torch.einsum("ri,bri->br", self.u_im, B))
        c_im = (torch.einsum("ri,bri->br", self.u_re, B)
                - torch.einsum("ri,bri->br", self.u_im, A))
        c = torch.cat([c_re, c_im], dim=-1)   # (B, 2R)
        c = self.act(c)
        return self.mix(c)


class NeuralLowRankDecoder(nn.Module):
    """Symmetric to NeuralLowRankEncoder: FiLM-modulated rank embeddings →
    rank-1 outer-product factors (d_re, d_im, a_re, a_im) → Hermitian sum.

    Replaces v5's `LowRankDecoder` which uses a single Linear(code_dim, R·128)
    to produce all rank-1 factors at once. The factored form is much cheaper
    at high R: cost grows as R·D·(H+W) instead of code_dim·R·(H+W).

    Each rank r owns a learned embedding F_r ∈ R^D. Codeword z provides FiLM
    γ(z), β(z) to modulate F_r per sample. Factor heads W_d_re, W_d_im,
    W_a_re, W_a_im decode the modulated embedding into the four real factors.

    FLOPs (cr=4, R=64, D=32):
        code_film 32K + 4·R·D·H 260K + 4 outer 260K  ≈ 0.55 M
    vs LowRankDecoder R=64 Linear:
        Linear(512, 64·128) = 4.2 M    (7.6× more)
    """

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 ranks: int = 16, d_emb: int = 32):
        super().__init__()
        self.h, self.w, self.ranks = h, w, ranks
        self.d_emb = d_emb

        # V2.5: direct per-rank features from z + PER-RANK factor heads.
        # Each rank has its own (D, H/W) projection so rank-1 outer products
        # are not confined to a shared H-subspace (which limited V2).
        self.code_proj = nn.Linear(code_dim, ranks * d_emb)
        nn.init.trunc_normal_(self.code_proj.weight, std=0.02)
        nn.init.zeros_(self.code_proj.bias)

        # Per-rank factor heads (R, D, H/W).
        self.W_d_re = nn.Parameter(torch.empty(ranks, d_emb, h))
        self.W_d_im = nn.Parameter(torch.empty(ranks, d_emb, h))
        self.W_a_re = nn.Parameter(torch.empty(ranks, d_emb, w))
        self.W_a_im = nn.Parameter(torch.empty(ranks, d_emb, w))
        for W in (self.W_d_re, self.W_d_im, self.W_a_re, self.W_a_im):
            nn.init.trunc_normal_(W, std=d_emb ** -0.5)

        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, code_dim) -> per-rank features (B, R, D)
        F_per_sample = self.code_proj(z).view(-1, self.ranks, self.d_emb)
        d_re = torch.einsum("brd,rdh->brh", F_per_sample, self.W_d_re)
        d_im = torch.einsum("brd,rdh->brh", F_per_sample, self.W_d_im)
        a_re = torch.einsum("brd,rdw->brw", F_per_sample, self.W_a_re)
        a_im = torch.einsum("brd,rdw->brw", F_per_sample, self.W_a_im)
        out_re = (torch.einsum("brh,brw->bhw", d_re, a_re)
                  + torch.einsum("brh,brw->bhw", d_im, a_im))
        out_im = (torch.einsum("brh,brw->bhw", d_im, a_re)
                  - torch.einsum("brh,brw->bhw", d_re, a_im))
        out = torch.stack([out_re, out_im], dim=1)
        out = out * self.scale.view(1, 2, 1, 1) + self.bias.view(1, 2, 1, 1)
        out = torch.tanh(out)
        return (out + 1.0) * 0.5


class GatedFactoredResidualDecoder(nn.Module):
    """Cheap residual rank bank: z → extra rank-1 components, per-rank gated.

    Output: alpha · Σ_r gate_r(z) · d_r(z) a_r(z)^H   (complex outer product).

    Used as a residual on top of a direct LowRankDecoder (see
    HybridRankDecoder). At init gate ≈ sigmoid(gate_bias) and alpha is small,
    so the bank starts contributing ~0 — model behaves like the direct
    decoder alone — and learns to open extra ranks where useful.
    """

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 extra_ranks: int = 64, d_emb: int = 16,
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
        F_per = self.code_proj(z).view(B, self.extra_ranks, self.d_emb)
        d_re = torch.einsum("brd,rdh->brh", F_per, self.W_d_re)
        d_im = torch.einsum("brd,rdh->brh", F_per, self.W_d_im)
        a_re = torch.einsum("brd,rdw->brw", F_per, self.W_a_re)
        a_im = torch.einsum("brd,rdw->brw", F_per, self.W_a_im)
        out_re = (torch.einsum("brh,brw->brhw", d_re, a_re)
                  + torch.einsum("brh,brw->brhw", d_im, a_im))
        out_im = (torch.einsum("brh,brw->brhw", d_im, a_re)
                  - torch.einsum("brh,brw->brhw", d_re, a_im))
        rank_out = torch.stack([out_re, out_im], dim=2)        # (B, R, 2, H, W)
        gate = torch.sigmoid(self.rank_gate(z)).view(B, self.extra_ranks, 1, 1, 1)
        residual = (gate * rank_out).sum(dim=1)                # (B, 2, H, W)
        return self.alpha * residual


class HybridRankDecoder(nn.Module):
    """Direct LowRankDecoder (R_d, fixed basis) + GatedFactoredResidualDecoder
    (R_e, z-conditioned cheap ranks).

    Cost-neutral capacity boost vs raw LowRankDecoder(R=R_d+R_e): factored
    branch is ~4× cheaper per rank, so total FLOPs ≈ direct(R_d) + 0.25·R_e.
    At init residual ≈ 0 (gate sigmoid≈0.12, alpha=1e-2) so the model starts
    at direct(R_d) and only opens extra ranks where they help.
    """

    def __init__(self, code_dim: int, h: int = 32, w: int = 32,
                 direct_ranks: int = 16, extra_ranks: int = 64,
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


class RankTokenAttention(nn.Module):
    """Treat the R bilinear measurements as a token sequence and let them
    self-attend.

    Input:  c ∈ (B, C, R)   — same shape as the bilinear output before mix
    Output: c + Δc          — residual, zero-init out-proj so starts at identity

    Each block is a standard pre-LN Transformer (MHA + FFN). Two blocks
    at d_model=32, num_heads=4 add ~1 M FLOPs for R=128 — tiny vs the
    rest of v9.
    """

    def __init__(self, in_channels: int = 2, r_enc: int = 128,
                 d_model: int = 32, num_heads: int = 4, num_blocks: int = 2,
                 ffn_mult: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.r_enc = r_enc
        self.d_model = d_model

        self.in_proj = nn.Linear(in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.empty(1, r_enc, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * ffn_mult),
                    nn.GELU(),
                    nn.Linear(d_model * ffn_mult, d_model),
                ),
            }))

        self.out_proj = nn.Linear(d_model, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: (B, C, R)
        z = c.transpose(1, 2)                # (B, R, C)
        z = self.in_proj(z) + self.pos_emb   # (B, R, D)
        for blk in self.blocks:
            h = blk["ln1"](z)
            attn_out, _ = blk["attn"](h, h, h, need_weights=False)
            z = z + attn_out
            z = z + blk["ffn"](blk["ln2"](z))
        delta = self.out_proj(z).transpose(1, 2)   # (B, C, R)
        return c + delta


class NeuralLowRankEncoder(nn.Module):
    """Input-adaptive low-rank bilinear measurement.

    Generates the rank-1 factors (u_r, v_r) from a learnable embedding
    matrix E ∈ (R, D) modulated by a per-sample FiLM that comes from a
    small context CNN. Then the bilinear measurement c[b,c,r] =
    u_r(x)^T H[b,c] v_r(x) is computed via two einsums (same shape as
    v5's static encoder).
    """

    def __init__(
        self,
        in_channels: int,
        h: int,
        w: int,
        code_dim: int,
        r_enc: int = 512,
        d_emb: int = 64,
        d_ctx: int = 32,
        nonlinearity: bool = True,
        rank_attn: RankTokenAttention | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.h, self.w = h, w
        self.r_enc, self.d_emb, self.d_ctx = r_enc, d_emb, d_ctx

        ctx_mid = max(d_ctx // 2, 8)
        self.context = nn.Sequential(
            ConvBN(in_channels, ctx_mid, 5, stride=2),
            nn.GELU(),
            ConvBN(ctx_mid, d_ctx, 5, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(d_ctx),
        )

        self.film = nn.Linear(d_ctx, 2 * d_emb)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.E = nn.Parameter(torch.empty(r_enc, d_emb))
        nn.init.trunc_normal_(self.E, std=d_emb ** -0.5)
        self.W_U = nn.Parameter(torch.empty(d_emb, h))
        self.W_V = nn.Parameter(torch.empty(d_emb, w))
        nn.init.trunc_normal_(self.W_U, std=d_emb ** -0.5)
        nn.init.trunc_normal_(self.W_V, std=d_emb ** -0.5)

        self.act = nn.GELU() if nonlinearity else nn.Identity()
        self.rank_attn = rank_attn  # optional RankTokenAttention; None disables
        self.mix = nn.Linear(in_channels * r_enc, code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.context(x)                                          # (B, D_ctx)
        gd, be = self.film(f).chunk(2, dim=-1)                       # (B, D_emb) each
        gamma = 1.0 + gd
        E_mod = gamma.unsqueeze(1) * self.E.unsqueeze(0) + be.unsqueeze(1)
        U = E_mod @ self.W_U                                         # (B, R, H)
        V = E_mod @ self.W_V                                         # (B, R, W)
        Hv = torch.einsum("bcij,brj->bcir", x, V)                    # (B, C, H, R)
        c = torch.einsum("bri,bcir->bcr", U, Hv)                     # (B, C, R)
        if self.rank_attn is not None:
            c = self.rank_attn(c)                                    # residual self-attn over R
        c = self.act(c).flatten(1)
        return self.mix(c)


class DilateDecoderPath(nn.Module):
    """Symmetric counterpart to ``DilateEncoderPath``.

    Codeword z → Linear(code_dim, C·H·W) → reshape → DCRDecoderBlock-style
    feature map → small gated residual. Linear weight is zero-init and a
    learnable scalar gate is ~0 at start, so the decoder begins using only
    the low-rank branch (H_a) and gradually opens this dilate branch.
    """

    def __init__(self, code_dim: int, in_channels: int = 2,
                 h: int = 32, w: int = 32, width: int = 8):
        super().__init__()
        self.in_channels, self.h, self.w = in_channels, h, w
        self.proj = nn.Linear(code_dim, in_channels * h * w)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.block = DCRDecoderRefine(width=width)
        self.gate = nn.Parameter(torch.tensor(1e-3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(-1, self.in_channels, self.h, self.w)
        return self.gate * self.block(x)


class EncoderFuseLinear(nn.Module):
    """Single Linear(2·code_dim → code_dim) fusion, identity-init.

    Drops the MLP bottleneck (hidden=64 was too narrow) and the residual
    z_a + z_b workaround. Identity init means output == z_a + z_b at step 0
    (= v8 sum behavior), and training learns full per-position cross-mixing.
    """

    def __init__(self, code_dim: int):
        super().__init__()
        self.code_dim = code_dim
        self.fuse = nn.Linear(2 * code_dim, code_dim)
        with torch.no_grad():
            self.fuse.weight.zero_()
            eye = torch.eye(code_dim)
            # weight[:, :code_dim] = I  → coefficient for z_a
            # weight[:, code_dim:] = I  → coefficient for z_b
            self.fuse.weight[:, :code_dim].copy_(eye)
            self.fuse.weight[:, code_dim:].copy_(eye)
            self.fuse.bias.zero_()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([z_a, z_b], dim=-1))


class EncoderFuseMLP(nn.Module):
    """Concat + 2-layer MLP fusion for encoder branches.

    Input: z_a, z_b ∈ (B, code_dim)  → concat (B, 2·code_dim) → MLP residual
    on top of (z_a + z_b)/2. fc2 is zero-init so output starts at v8 sum
    behavior, and learns richer fusion over time.
    """

    def __init__(self, code_dim: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = max(code_dim // 4, 64)
        self.hidden = hidden
        self.fc1 = nn.Linear(2 * code_dim, hidden)
        self.fc2 = nn.Linear(hidden, code_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_a, z_b], dim=-1)
        h = torch.nn.functional.gelu(self.fc1(h))
        h = self.fc2(h)
        return z_a + z_b + h


class EncoderFuse1x1(nn.Module):
    """Fuse two parallel codeword branches via small 1x1 conv MLP.

    Replaces v8's `z_a + gate * z_b` with a learnable per-position non-linear
    mixing (still position-shared via Conv1d kernel=1). Stack the two
    branches as channels: (B, 2, code_dim) → Conv1d(2→h, 1) → GELU →
    Conv1d(h, 1, 1) → (B, 1, code_dim) → squeeze.

    Init: identity-like (output averages the two branches).
    """

    def __init__(self, hidden: int = 4):
        super().__init__()
        self.mix1 = nn.Conv1d(2, hidden, kernel_size=1)
        self.mix2 = nn.Conv1d(hidden, 1, kernel_size=1)
        # Init mix1 so each hidden unit ≈ (z_a + z_b)/2; mix2 averages.
        with torch.no_grad():
            self.mix1.weight.fill_(0.5)
            self.mix1.bias.zero_()
            self.mix2.weight.fill_(1.0 / hidden)
            self.mix2.bias.zero_()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z = torch.stack([z_a, z_b], dim=1)         # (B, 2, code_dim)
        z = self.mix1(z)                            # (B, h, code_dim)
        z = torch.nn.functional.gelu(z)
        z = self.mix2(z).squeeze(1)                 # (B, code_dim)
        return z


class DecoderFuse1x1(nn.Module):
    """Fuse two parallel decoder branches (H_a, H_b ∈ R^{B,2,H,W})
    via 1x1 Conv2d on stacked channels (4 → 2).

    Init: identity for H_a (passthrough), zero for H_b — model starts
    using only the low-rank branch.
    """

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0, 0] = 1.0   # out_re = H_a_re
            self.conv.weight[1, 1, 0, 0] = 1.0   # out_im = H_a_im
            self.conv.bias.zero_()

    def forward(self, H_a: torch.Tensor, H_b: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([H_a, H_b], dim=1))


class IterativeRefine(nn.Module):
    """K stacked ``DCRDecoderRefine`` stages.

    Each stage is residual with a learnable scalar gate initialized
    near zero, so the stack starts at identity and only learns to
    refine on top of the rank-R coarse reconstruction. The refine
    block already shrinks to a (2, H, W) tensor and is shape-preserving
    end-to-end.
    """

    def __init__(self, width: int = 8, K: int = 2):
        super().__init__()
        assert K >= 1
        self.stages = nn.ModuleList([DCRDecoderRefine(width=width) for _ in range(K)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for s in self.stages:
            x = s(x)
        return x


class DCRNetV9(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        reduction: int = 4,
        expansion: int = 1,        # kept for CLI symmetry
        r_enc: int = 512,
        ranks: int = 16,
        d_emb: int = 64,
        d_ctx: int = 32,
        refine_width: int = 8,
        refine_K: int = 2,
        use_dilate_path: bool = True,
        nonlinearity: bool = True,
        rank_attn_blocks: int = 0,
        rank_attn_dim: int = 32,
        rank_attn_heads: int = 4,
        use_film: bool = True,
        neural_decoder: bool = False,
        dec_d_emb: int = 32,
        hybrid_decoder: bool = False,
        hybrid_direct_ranks: int = 16,
        hybrid_extra_ranks: int = 64,
        hybrid_extra_d_emb: int = 16,
        hybrid_gate_bias: float = -2.0,
        hybrid_alpha_init: float = 1e-2,
        enc_fuse_1x1: bool = False,
        enc_fuse_mlp: int = 0,        # 0 disables; else hidden dim for concat+MLP fuse
        enc_fuse_linear: bool = False,  # single Linear(2D→D) identity-init fusion
        dec_dilate: bool = False,
        dec_dilate_width: int = 8,
        complex_encoder: bool = False,  # use true complex Hermitian bilinear encoder
        h: int = 32,
        w: int = 32,
        total_size: int | None = None,
    ):
        super().__init__()
        if total_size is None:
            total_size = 2 * h * w
        code_dim = total_size // reduction
        del expansion

        if complex_encoder:
            # True complex Hermitian bilinear (4 cross-terms preserved)
            assert rank_attn_blocks == 0, "rank_attn not supported with complex_encoder"
            assert in_channels == 2, "complex_encoder requires in_channels=2 (re, im)"
            self.enc_neural = ComplexLowRankEncoder(
                h, w, code_dim, r_enc=r_enc, nonlinearity=nonlinearity,
            )
        elif use_film:
            rank_attn = None
            if rank_attn_blocks > 0:
                rank_attn = RankTokenAttention(
                    in_channels=in_channels,
                    r_enc=r_enc,
                    d_model=rank_attn_dim,
                    num_heads=rank_attn_heads,
                    num_blocks=rank_attn_blocks,
                )
            self.enc_neural = NeuralLowRankEncoder(
                in_channels, h, w, code_dim,
                r_enc=r_enc, d_emb=d_emb, d_ctx=d_ctx,
                nonlinearity=nonlinearity,
                rank_attn=rank_attn,
            )
        else:
            # v8 / v5-style static encoder: direct u, v parameters (no FiLM, no rank
            # bottleneck through D-dim embeddings). Strictly more expressive at same R.
            assert rank_attn_blocks == 0, "rank_attn requires FiLM encoder path"
            self.enc_neural = LowRankEncoder(
                in_channels, h, w, code_dim,
                r_enc=r_enc, nonlinearity=nonlinearity,
            )
        self.use_film = use_film
        self.complex_encoder = complex_encoder

        self.use_dilate_path = use_dilate_path
        if use_dilate_path:
            self.enc_dilate = DilateEncoderPath(in_channels, h, w, code_dim)

        # Encoder fusion alternatives (vs default sum z_a + z_b)
        self.enc_fuse_1x1 = enc_fuse_1x1 and use_dilate_path
        self.enc_fuse_mlp_dim = enc_fuse_mlp if use_dilate_path else 0
        self.enc_fuse_linear = enc_fuse_linear and use_dilate_path
        if self.enc_fuse_1x1:
            self.fuse_enc = EncoderFuse1x1(hidden=4)
        elif self.enc_fuse_mlp_dim > 0:
            self.fuse_enc = EncoderFuseMLP(code_dim, hidden=self.enc_fuse_mlp_dim)
        elif self.enc_fuse_linear:
            self.fuse_enc = EncoderFuseLinear(code_dim)

        if hybrid_decoder:
            assert not neural_decoder, "hybrid_decoder and neural_decoder are mutually exclusive"
            self.decoder = HybridRankDecoder(
                code_dim, h, w,
                direct_ranks=hybrid_direct_ranks,
                extra_ranks=hybrid_extra_ranks,
                extra_d_emb=hybrid_extra_d_emb,
                gate_bias=hybrid_gate_bias,
                alpha_init=hybrid_alpha_init,
            )
        elif neural_decoder:
            self.decoder = NeuralLowRankDecoder(code_dim, h, w, ranks=ranks, d_emb=dec_d_emb)
        else:
            self.decoder = LowRankDecoder(code_dim, h, w, ranks=ranks)

        # Decoder dilate parallel branch (z → dense reconstruction, fused with H_a)
        self.dec_dilate = dec_dilate
        if dec_dilate:
            self.decoder_dilate = DilateDecoderPath(
                code_dim, in_channels=in_channels, h=h, w=w, width=dec_dilate_width)
            self.fuse_dec = DecoderFuse1x1(in_channels=in_channels)

        self.refine = IterativeRefine(width=refine_width, K=refine_K)

        # Optional SA + SE blocks (CLNet-style). Identity-init: SA mask ≈ 1,
        # SE gain ≈ 1, so warm-start from existing ckpt is non-disruptive.
        # Gated by env vars DCRNET_V9_SA=1, DCRNET_V9_SE=1.
        # Additional placements (more impactful per CLNet logic):
        #   DCRNET_V9_SA_ENC=1 — SA inside enc_dilate (see DilateEncoderPath)
        #   DCRNET_V9_SE_Z=1 — SE on codeword z (per-rank attention)
        import os as _os
        self._use_sa = _os.environ.get('DCRNET_V9_SA', '0') == '1'
        self._use_se = _os.environ.get('DCRNET_V9_SE', '0') == '1'
        self._use_se_z = _os.environ.get('DCRNET_V9_SE_Z', '0') == '1'
        if self._use_se_z:
            bn = max(code_dim // 16, 4)
            self.se_z_fc1 = nn.Linear(code_dim, bn, bias=True)
            self.se_z_fc2 = nn.Linear(bn, code_dim, bias=True)
            nn.init.zeros_(self.se_z_fc1.weight)
            nn.init.zeros_(self.se_z_fc1.bias)
            nn.init.zeros_(self.se_z_fc2.weight)
            nn.init.zeros_(self.se_z_fc2.bias)
        if self._use_sa:
            # 7x7 conv on (max, avg) channel-pool of (B, 2, H, W) → 1-ch logits
            self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=True)
            nn.init.zeros_(self.sa_conv.weight)
            nn.init.constant_(self.sa_conv.bias, 3.0)   # sigmoid(3) ≈ 0.953 ≈ 1
        if self._use_se:
            # SE on the codeword channels — works on the coarse pre-refine
            # output (B, 2, H, W) via global pool. 2 → 1 → 2.
            self.se_pool = nn.AdaptiveAvgPool2d(1)
            self.se_fc1 = nn.Linear(2, 1, bias=False)
            self.se_fc2 = nn.Linear(1, 2, bias=False)
            nn.init.zeros_(self.se_fc1.weight)
            nn.init.zeros_(self.se_fc2.weight)   # 0-init → sigmoid(0)*2 ≈ 1

    def _apply_sa_se(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_se:
            b, c, _, _ = x.shape
            s = self.se_pool(x).view(b, c)
            s = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(s)))) * 2.0
            x = x * s.view(b, c, 1, 1)
        if self._use_sa:
            mask_in = torch.cat([x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], dim=1)
            mask = torch.sigmoid(self.sa_conv(mask_in))
            x = x * mask
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x * 2.0 - 1.0
        z_a = self.enc_neural(x_centered)
        if self.use_dilate_path:
            z_b = self.enc_dilate(x)
            if self.enc_fuse_1x1 or self.enc_fuse_mlp_dim > 0 or self.enc_fuse_linear:
                z = self.fuse_enc(z_a, z_b)
            else:
                z = z_a + z_b
        else:
            z = z_a
        # Optional SE on the codeword z (per-rank attention).
        # DCRNET_V9_SE_Z=1 enables. Identity-init: weight=0, bias=0,
        # sigmoid(0)*2 = 1 → no gain change at start, then learns.
        if self._use_se_z:
            b = z.shape[0]
            gain = torch.sigmoid(self.se_z_fc2(F.relu(self.se_z_fc1(z)))) * 2.0
            z = z * gain
        return z

    def _coarse_from_z(self, z: torch.Tensor) -> torch.Tensor:
        H_a = self.decoder(z)
        if self.dec_dilate:
            H_b = self.decoder_dilate(z)
            return self.fuse_dec(H_a, H_b)
        return H_a

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        refined = self._apply_sa_se(refined)
        if clamp:
            # DCRNET_V9_HSIGMOID=1 swaps hard clamp for hsigmoid: identical to
            # clamp(x, 0, 1) inside [0, 1] but produces soft gradient outside,
            # so a warm-start from a clamp-trained checkpoint just adds
            # gradient signal at the boundaries without changing in-range output.
            import os as _os
            if _os.environ.get('DCRNET_V9_HSIGMOID', '0') == '1':
                import torch.nn.functional as _F
                return _F.relu6(refined * 6.0) / 6.0
            return refined.clamp(0.0, 1.0)
        return refined

    def decode_with_coarse(self, z: torch.Tensor, clamp: bool = True):
        """Return (refined, coarse), optionally clamped to [0,1]."""
        coarse = self._coarse_from_z(z)
        refined = self.refine(coarse)
        if clamp:
            return refined.clamp(0.0, 1.0), coarse.clamp(0.0, 1.0)
        return refined, coarse

    def forward(self, x: torch.Tensor, return_coarse: bool = False, clamp: bool = True):
        z = self.encode(x)
        if return_coarse:
            return self.decode_with_coarse(z, clamp=clamp)
        return self.decode(z, clamp=clamp)


def dcrnet_v9(
    reduction: int = 4,
    expansion: int = 1,
    ranks: int = 16,
    r_enc: int = 512,
    h: int = 32,
    w: int = 32,
    total_size: int | None = None,
    d_emb: int = 64,
    d_ctx: int = 32,
    refine_width: int = 8,
    refine_K: int = 2,
    use_dilate_path: bool = True,
    rank_attn_blocks: int = 0,
    rank_attn_dim: int = 32,
    rank_attn_heads: int = 4,
    use_film: bool = True,
    neural_decoder: bool = False,
    dec_d_emb: int = 32,
    hybrid_decoder: bool = False,
    hybrid_direct_ranks: int = 16,
    hybrid_extra_ranks: int = 64,
    hybrid_extra_d_emb: int = 16,
    hybrid_gate_bias: float = -2.0,
    hybrid_alpha_init: float = 1e-2,
    enc_fuse_1x1: bool = False,
    enc_fuse_mlp: int = 0,
    enc_fuse_linear: bool = False,
    dec_dilate: bool = False,
    dec_dilate_width: int = 8,
    complex_encoder: bool = False,
    **kwargs,
) -> DCRNetV9:
    return DCRNetV9(
        reduction=reduction,
        expansion=expansion,
        ranks=ranks,
        r_enc=r_enc,
        d_emb=d_emb,
        d_ctx=d_ctx,
        refine_width=refine_width,
        refine_K=refine_K,
        use_dilate_path=use_dilate_path,
        rank_attn_blocks=rank_attn_blocks,
        rank_attn_dim=rank_attn_dim,
        rank_attn_heads=rank_attn_heads,
        use_film=use_film,
        neural_decoder=neural_decoder,
        dec_d_emb=dec_d_emb,
        hybrid_decoder=hybrid_decoder,
        hybrid_direct_ranks=hybrid_direct_ranks,
        hybrid_extra_ranks=hybrid_extra_ranks,
        hybrid_extra_d_emb=hybrid_extra_d_emb,
        hybrid_gate_bias=hybrid_gate_bias,
        hybrid_alpha_init=hybrid_alpha_init,
        enc_fuse_1x1=enc_fuse_1x1,
        enc_fuse_mlp=enc_fuse_mlp,
        enc_fuse_linear=enc_fuse_linear,
        dec_dilate=dec_dilate,
        dec_dilate_width=dec_dilate_width,
        complex_encoder=complex_encoder,
        h=h,
        w=w,
        total_size=total_size,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cr", type=int, default=4)
    p.add_argument("--ranks", type=int, default=16)
    p.add_argument("--r-enc", type=int, default=512)
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--refine-width", type=int, default=8)
    p.add_argument("--refine-K", type=int, default=2)
    p.add_argument("--no-dilate", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = dcrnet_v9(
        reduction=args.cr,
        ranks=args.ranks,
        r_enc=args.r_enc,
        d_emb=args.d_emb,
        refine_width=args.refine_width,
        refine_K=args.refine_K,
        use_dilate_path=not args.no_dilate,
    ).to(device)
    x = torch.rand(2, 2, 32, 32, device=device)
    z = m.encode(x)
    y = m.decode(z)
    n_total = sum(p.numel() for p in m.parameters())
    n_enc_n = sum(p.numel() for p in m.enc_neural.parameters())
    n_enc_d = sum(p.numel() for p in m.enc_dilate.parameters()) if m.use_dilate_path else 0
    n_dec = sum(p.numel() for p in m.decoder.parameters())
    n_ref = sum(p.numel() for p in m.refine.parameters())
    print(
        f"cr={args.cr} ranks={args.ranks} r_enc={args.r_enc} d_emb={args.d_emb} "
        f"refine_w={args.refine_width} K={args.refine_K} dilate={not args.no_dilate}"
    )
    print(f"codeword={tuple(z.shape)}  out={tuple(y.shape)}")
    print(
        f"total={n_total/1e3:.1f}K  enc_neural={n_enc_n/1e3:.1f}K  "
        f"enc_dilate={n_enc_d/1e3:.1f}K  dec={n_dec/1e3:.1f}K  refine={n_ref/1e3:.1f}K"
    )
