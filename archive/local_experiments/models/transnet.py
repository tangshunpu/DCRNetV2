"""TransNet — Transformer-based CSI Feedback.

Ported from https://github.com/Treedy2020/TransNet
Reference: Cui, Y., et al., "TransNet: Full Attention Network for CSI Feedback
in FDD Massive MIMO," IEEE Wireless Commun. Lett., 2022.

Differences from upstream source:
    - dropped `from utils import logger`
    - h, w, total_size are kwargs, default 32×32 / 2048
    - Uses PyTorch's nn.MultiheadAttention instead of the upstream's custom
      reimplementation (mathematically equivalent, drops ~150 lines of code)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["transnet", "Transformer"]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 batch_first=False, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src, need_weights=False)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 batch_first=False, layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2, _ = self.multihead_attn(tgt, memory, memory, need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model: int = 64, nhead: int = 2,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 dim_feedforward: int = 2048, dropout: float = 0.0,
                 reduction: int = 4, h: int = 32, w: int = 32,
                 total_size: Optional[int] = None, in_channel: int = 2,
                 **_unused):
        super().__init__()
        if total_size is None:
            total_size = in_channel * h * w
        assert total_size % d_model == 0, (
            f'd_model={d_model} must divide total_size={total_size}')
        self.feature_shape = (total_size // d_model, d_model)
        self.total_size = total_size
        self.in_channel, self.h, self.w = in_channel, h, w
        self.d_model = d_model
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        # Note: the upstream TransNet adds a LayerNorm here (decoder_norm).
        # With our added Sigmoid output it collapses the pre-Sigmoid signal
        # to std≈1e-3 (gradient driven down so output mean stays at 0.5),
        # killing learning. Each TransformerDecoderLayer already has 3
        # internal LayerNorms, so this final one is redundant.
        # self.decoder_norm = nn.LayerNorm(d_model, eps=1e-5)  # disabled
        self.fc_encoder = nn.Linear(total_size, total_size // reduction)
        self.fc_decoder = nn.Linear(total_size // reduction, total_size)
        # End-of-network Sigmoid: matches CSINet/CRNet/CLNet/DCRNet/v9 convention
        # of bounding output to [0,1]. Upstream TransNet code skipped it and
        # relied on long training to bring the unbounded LayerNorm output into
        # the target range — extremely slow to converge on large-input datasets
        # (e.g. Rice 2x96x52 stays at +30dB NMSE for hundreds of epochs).
        self.sigmoid = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (n, c, h, w) → flatten → (n, L=total_size/d_model, d_model)
        # nn.MultiheadAttention default expects (L, N, E) — so we transpose first.
        n = x.size(0)
        src = x.view(n, self.feature_shape[0], self.feature_shape[1])
        src = src.transpose(0, 1)                                    # (L, N, E)
        memory = src
        for layer in self.encoder:
            memory = layer(memory)
        memory = memory.transpose(0, 1).contiguous().view(n, -1)     # (N, L*E)
        codeword = self.fc_encoder(memory)
        rec = self.fc_decoder(codeword).view(n, self.feature_shape[0], self.feature_shape[1])
        rec = rec.transpose(0, 1)
        tgt = rec
        for layer in self.decoder:
            tgt = layer(tgt, rec)
        tgt = tgt.transpose(0, 1)                                    # (N, L, E)
        out = tgt.contiguous().view(n, self.in_channel, self.h, self.w)
        return self.sigmoid(out)


def transnet(reduction: int = 4, expansion: int = 1, d_model: int = 64,
             h: int = 32, w: int = 32, total_size: Optional[int] = None,
             **_unused):
    del expansion
    return Transformer(d_model=d_model, num_encoder_layers=2,
                       num_decoder_layers=2, nhead=2, reduction=reduction,
                       dropout=0.0, h=h, w=w, total_size=total_size)
