"""Training utilities for DCRNetV2: LR scheduler, meters, and metrics."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpCosineAnnealingLR(_LRScheduler):
    """Linear warmup over ``T_warmup`` steps, then cosine anneal to ``eta_min``."""

    def __init__(self, optimizer, T_max: int, T_warmup: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = max(1, int(T_max))
        self.T_warmup = max(0, int(T_warmup))
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_warmup > 0 and self.last_epoch < self.T_warmup:
            scale = max(1, self.last_epoch) / self.T_warmup
            return [base_lr * scale for base_lr in self.base_lrs]
        denom = max(1, self.T_max - self.T_warmup)
        progress = min(1.0, max(0.0, (self.last_epoch - self.T_warmup) / denom))
        k = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base_lr - self.eta_min) * k for base_lr in self.base_lrs]


class AverageMeter:
    """Running average accumulator."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


@torch.no_grad()
def nmse_ratio(sparse_pred: torch.Tensor, sparse_gt: torch.Tensor) -> torch.Tensor:
    """Return one linear NMSE ratio per angular-delay CSI sample."""

    sparse_gt = sparse_gt - 0.5
    sparse_pred = sparse_pred - 0.5
    power_gt = sparse_gt[:, 0] ** 2 + sparse_gt[:, 1] ** 2
    diff = sparse_gt - sparse_pred
    mse = diff[:, 0] ** 2 + diff[:, 1] ** 2
    return mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2]).clamp_min(1e-12)


@torch.no_grad()
def nmse_db(sparse_pred: torch.Tensor, sparse_gt: torch.Tensor) -> float:
    """NMSE in dB on angular-delay CSI tensors shaped ``(B, 2, 32, 32)``."""

    nmse = 10 * torch.log10(nmse_ratio(sparse_pred, sparse_gt).mean())
    return nmse.item()


@torch.no_grad()
def evaluator(sparse_pred: torch.Tensor, sparse_gt: torch.Tensor, raw_gt: torch.Tensor):
    """Return ``(rho, nmse_db)`` for COST2100.

    Args:
        sparse_pred: Reconstructed angular-delay tensor ``(B, 2, 32, 32)``.
        sparse_gt: Ground-truth angular-delay tensor ``(B, 2, 32, 32)``.
        raw_gt: Full-bandwidth complex CSI ``(B, 32, 125, 2)``.
    """

    nt, nc, nc_expand = 32, 32, 257
    nmse = nmse_db(sparse_pred, sparse_gt)

    sparse_pred = (sparse_pred - 0.5).permute(0, 2, 3, 1).contiguous()
    n = sparse_pred.size(0)
    zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
    sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
    raw_pred = torch.view_as_real(
        torch.fft.fft(torch.view_as_complex(sparse_pred.contiguous()), dim=-1)
    )[:, :, :125, :]

    norm_pred = torch.sqrt((raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2).sum(dim=1)).clamp_min(1e-12)
    norm_gt = torch.sqrt((raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2).sum(dim=1)).clamp_min(1e-12)
    real_cross = (raw_pred[..., 0] * raw_gt[..., 0] + raw_pred[..., 1] * raw_gt[..., 1]).sum(dim=1)
    imag_cross = (raw_pred[..., 0] * raw_gt[..., 1] - raw_pred[..., 1] * raw_gt[..., 0]).sum(dim=1)
    norm_cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)
    rho = (norm_cross / (norm_pred * norm_gt)).mean()

    return rho.item(), nmse
