"""Modern training recipe for DCRNet — 200 epochs.

Differences vs main.py:
  - AdamW with weight decay (1e-4)
  - Warmup-Stable-Decay (WSD) schedule: 5 ep linear warmup → stable plateau
    → linear decay to eta_min. No cosine — avoids the "lr stays high too long"
    pathology of CosineAnnealing where 80% of training happens above 50% lr.
  - Larger batch size (default 400 vs 200) — more stable gradients, fewer steps
  - Higher peak lr (default 3e-3 vs 2e-3) — compensate shorter schedule
  - Gradient clipping (max_norm=1.0) — prevents occasional spikes
  - Optional EMA model used for evaluation (default on)
  - Optional AMP mixed precision (default off — small models, marginal speedup)

Usage:
  python train_modern.py --gpu 0 --cr 4 --scenario in --model v8 \
      --expansion 4 --ranks 16 --r-enc 1024
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import thop
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import inspect
import models as _models
from dataset import Cost2100DataLoader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('DCRNet modern training (200 ep, AdamW, WSD)')

    # data / model
    p.add_argument('--data', default='./COST2100')
    p.add_argument('--scenario', default='in', choices=['in', 'out'])
    p.add_argument('--cr', type=int, default=4)
    p.add_argument('--model', default='v8', choices=['v1', 'v5', 'v8', 'v9'])
    p.add_argument('--expansion', type=int, default=1)
    p.add_argument('--ranks', type=int, default=None)
    p.add_argument('--r-enc', dest='r_enc', type=int, default=None)

    # training schedule
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--scheduler', choices=['wsd', 'reduce'], default='wsd',
                   help='wsd = warmup-stable-decay (default); '
                        'reduce = ReduceLROnPlateau on test NMSE (continuation-friendly)')
    p.add_argument('--warmup-epochs', dest='warmup_epochs', type=int, default=5)
    p.add_argument('--decay-epochs', dest='decay_epochs', type=int, default=50,
                   help='last N epochs do linear decay from peak_lr to eta_min (wsd only)')
    p.add_argument('--patience', type=int, default=3,
                   help='ReduceLROnPlateau: val-cycles without improvement before lr halved')
    p.add_argument('--factor', type=float, default=0.5,
                   help='ReduceLROnPlateau: lr *= factor on plateau')
    p.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=400)
    p.add_argument('--lr', '--learning-rate', dest='lr', type=float, default=3e-3,
                   help='peak learning rate (post-warmup)')
    p.add_argument('--eta-min', dest='eta_min', type=float, default=5e-5)
    p.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', dest='grad_clip', type=float, default=1.0)

    # modern training tricks
    p.add_argument('--ema', action='store_true', default=True,
                   help='use EMA weights for evaluation (default on)')
    p.add_argument('--no-ema', dest='ema', action='store_false')
    p.add_argument('--ema-decay', dest='ema_decay', type=float, default=0.999)
    p.add_argument('--amp', action='store_true', default=False,
                   help='use mixed precision (fp16) training')

    # warm-init from v5 (mirror main.py flags)
    p.add_argument('--init-from-v5', dest='init_from_v5', type=str, default=None)
    p.add_argument('--freeze-v5', dest='freeze_v5', action='store_true')
    p.add_argument('--backbone-lr', dest='backbone_lr', type=float, default=None)

    # misc
    p.add_argument('--gpu', default=None, type=str)
    p.add_argument('-j', '--workers', type=int, default=4)
    p.add_argument('--outputs', default='./outputs')
    p.add_argument('--val-freq', '-v', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--pretrained', type=str, default=None)
    return p


# ---------------------------------------------------------------------------
# Schedule: Warmup-Stable-Decay
# ---------------------------------------------------------------------------

def make_wsd_schedule(optimizer, total_steps: int, warmup_steps: int,
                      decay_steps: int, eta_min_ratio: float):
    """Linear warmup → stable at peak → linear decay to eta_min.

    eta_min_ratio = eta_min / peak_lr.  Returned LambdaLR multiplies base_lr
    by a factor in [eta_min_ratio, 1].
    """
    stable_end = total_steps - decay_steps
    assert warmup_steps < stable_end, "warmup must end before decay starts"

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < stable_end:
            return 1.0
        # linear decay over decay_steps
        frac = (step - stable_end) / max(1, decay_steps)
        return eta_min_ratio + (1 - eta_min_ratio) * (1 - frac)

    return LambdaLR(optimizer, lr_lambda=lr_factor)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        # also copy buffers (BN running stats)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


# ---------------------------------------------------------------------------
# Logging / utils
# ---------------------------------------------------------------------------

def setup_logger(filename: str) -> logging.Logger:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger = logging.getLogger('train_modern')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] %(message)s')
    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _einsum_macs(model, batch_size: int = 1) -> int:
    extra = 0
    for m in model.modules():
        cls = m.__class__.__name__
        if cls == 'LowRankEncoder':
            R, H, W = m.r_enc, m.h, m.w
            C = m.mix.in_features // R
            extra += batch_size * C * H * W * R + batch_size * C * H * R
        elif cls == 'LowRankDecoder':
            R, H, W = m.ranks, m.h, m.w
            extra += 4 * batch_size * R * H * W
        elif cls == 'AxialAttention':
            D, H, W = m.dim, m.h, m.w
            if m.axis == 'w':
                extra += 2 * batch_size * D * H * W * W
            else:
                extra += 2 * batch_size * D * H * H * W
    return extra


def load_model(args):
    factory_name = 'dcrnet' if args.model == 'v1' else f'dcrnet_{args.model}'
    factory = getattr(_models, factory_name)
    sig = inspect.signature(factory)
    kwargs = {'reduction': args.cr, 'expansion': args.expansion}
    if args.ranks is not None and 'ranks' in sig.parameters:
        kwargs['ranks'] = args.ranks
    if args.r_enc is not None and 'r_enc' in sig.parameters:
        kwargs['r_enc'] = args.r_enc
    model = factory(**kwargs)
    image = torch.randn([1, 2, 32, 32])
    flops_thop, _ = thop.profile(model, inputs=(image,), verbose=False)
    flops = flops_thop + _einsum_macs(model, batch_size=1)
    params = sum(p.numel() for p in model.parameters())
    return model, flops, params


def evaluator(sparse_pred, sparse_gt, raw_gt):
    """NMSE + ρ. Same as main.py."""
    with torch.no_grad():
        nt, nc, nc_expand = 32, 32, 257
        sparse_gt = sparse_gt - 0.5
        sparse_pred = sparse_pred - 0.5
        power_gt = sparse_gt[:, 0] ** 2 + sparse_gt[:, 1] ** 2
        diff = sparse_gt - sparse_pred
        mse = diff[:, 0] ** 2 + diff[:, 1] ** 2
        nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())

        n = sparse_pred.size(0)
        sparse_pred = sparse_pred.permute(0, 2, 3, 1)
        zeros = sparse_pred.new_zeros((n, nt, nc_expand - nc, 2))
        sparse_pred = torch.cat((sparse_pred, zeros), dim=2)
        raw_pred = torch.view_as_real(
            torch.fft.fft(torch.view_as_complex(sparse_pred.contiguous()), dim=-1)
        )[:, :, :125, :]
        norm_pred = torch.sqrt(
            (raw_pred[..., 0] ** 2 + raw_pred[..., 1] ** 2).sum(dim=1))
        norm_gt = torch.sqrt(
            (raw_gt[..., 0] ** 2 + raw_gt[..., 1] ** 2).sum(dim=1))
        real_cross = (raw_pred[..., 0] * raw_gt[..., 0]
                      + raw_pred[..., 1] * raw_gt[..., 1]).sum(dim=1)
        imag_cross = (raw_pred[..., 0] * raw_gt[..., 1]
                      - raw_pred[..., 1] * raw_gt[..., 0]).sum(dim=1)
        cross = torch.sqrt(real_cross ** 2 + imag_cross ** 2)
        rho = (cross / (norm_pred * norm_gt + 1e-12)).mean()
        return rho, nmse


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device,
                    grad_clip, scaler, ema, epoch, total_epochs,
                    step_scheduler_each_batch: bool = True):
    model.train()
    losses, n = 0.0, 0
    use_amp = scaler is not None
    bar = tqdm(loader, file=sys.stdout, ncols=120)
    for (sparse_gt,) in bar:
        sparse_gt = sparse_gt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                pred = model(sparse_gt)
                loss = criterion(pred, sparse_gt)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(sparse_gt)
            loss = criterion(pred, sparse_gt)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if step_scheduler_each_batch:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        losses += loss.item() * sparse_gt.size(0)
        n += sparse_gt.size(0)
        bar.set_description(f'ep[{epoch + 1}/{total_epochs}]')
        # ReduceLROnPlateau lacks get_last_lr() — read from optimizer.param_groups
        cur_lr = optimizer.param_groups[0]['lr']
        bar.set_postfix(lr=f'{cur_lr:.2e}',
                        loss=f'{losses / n:.3e}')
    return losses / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rho_sum, nmse_sum, n = 0.0, 0.0, 0
    for sparse_gt, raw_gt in loader:
        sparse_gt = sparse_gt.to(device, non_blocking=True)
        pred = model(sparse_gt)
        rho, nmse = evaluator(pred, sparse_gt, raw_gt)
        bs = sparse_gt.size(0)
        rho_sum += rho.item() * bs
        nmse_sum += nmse.item() * bs
        n += bs
    return nmse_sum / n, rho_sum / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    # device
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    # logger / outputs
    tag = (f'{time.strftime("%m%d%H%M")}-modern-{args.model}-'
           f'{args.expansion}X-{args.cr}-{args.scenario}')
    log_path = os.path.join(args.outputs, 'log', f'{tag}.log')
    ckpt_dir = os.path.join(args.outputs, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(log_path)

    # model
    model, flops, params = load_model(args)
    logger.info(f'Model: DCRNet-{args.model}  '
                f'cr=1/{args.cr} expansion={args.expansion} '
                f'ranks={args.ranks} r_enc={args.r_enc}')
    logger.info(f'FLOPs: {flops/1e6:.3f}M  params: {params/1e3:.1f}K')
    model = model.to(device)

    # warm-init from v5 (optional)
    if args.init_from_v5:
        v5_sd = torch.load(args.init_from_v5, map_location=device)['state_dict']
        remap = {}
        for k, v in v5_sd.items():
            if 'total_ops' in k or 'total_params' in k:
                continue
            if k.startswith('encoder.'):
                remap['enc_lowrank.' + k[len('encoder.'):]] = v
            elif k.startswith('decoder.'):
                remap[k] = v
        model.load_state_dict(remap, strict=False)
        if args.freeze_v5:
            for n, p in model.named_parameters():
                if n.startswith('enc_lowrank.') or n.startswith('decoder.'):
                    p.requires_grad = False
        logger.info(f'init from v5: {args.init_from_v5}  freeze={args.freeze_v5}')

    # load pretrained weights (continue-training style — fresh optimizer/scheduler)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f'pretrained: {args.pretrained}  '
                    f'(missing={len(missing)}, unexpected={len(unexpected)}, '
                    f"prev_best={ckpt.get('best_nmse', 'n/a')})")

    # data
    pin_mem = device == 'cuda'
    train_loader, _, test_loader = Cost2100DataLoader(
        root=args.data, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=pin_mem,
        scenario=args.scenario)()
    steps_per_ep = len(train_loader)
    total_steps = args.epochs * steps_per_ep
    warmup_steps = args.warmup_epochs * steps_per_ep
    decay_steps = args.decay_epochs * steps_per_ep
    logger.info(f'batch_size={args.batch_size}  steps/ep={steps_per_ep}  '
                f'total_steps={total_steps}')
    logger.info(f'schedule: warmup={args.warmup_epochs}ep  '
                f'stable={args.epochs - args.warmup_epochs - args.decay_epochs}ep  '
                f'decay={args.decay_epochs}ep  '
                f'peak_lr={args.lr}  eta_min={args.eta_min}')

    # optimizer (AdamW + optional param groups)
    if args.init_from_v5 and args.backbone_lr is not None and not args.freeze_v5:
        backbone_params, new_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if (n.startswith('enc_lowrank.') or n.startswith('decoder.'))
             else new_params).append(p)
        optimizer = torch.optim.AdamW(
            [{'params': backbone_params, 'lr': args.backbone_lr},
             {'params': new_params,      'lr': args.lr}],
            weight_decay=args.weight_decay, betas=(0.9, 0.999))
        logger.info(f'param groups: backbone lr={args.backbone_lr} | new lr={args.lr}')
    else:
        params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params_list, lr=args.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)

    if args.scheduler == 'wsd':
        eta_min_ratio = args.eta_min / args.lr
        scheduler = make_wsd_schedule(optimizer, total_steps, warmup_steps,
                                      decay_steps, eta_min_ratio)
        step_each_batch = True
        logger.info(f'scheduler=wsd  warmup={args.warmup_epochs}ep '
                    f'stable={args.epochs - args.warmup_epochs - args.decay_epochs}ep '
                    f'decay={args.decay_epochs}ep')
    else:  # 'reduce' — ReduceLROnPlateau, stepped per eval
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=args.factor,
            patience=args.patience, min_lr=args.eta_min,
        )
        step_each_batch = False
        logger.info(f'scheduler=reduce  patience={args.patience} '
                    f'factor={args.factor} min_lr={args.eta_min} '
                    f'(steps on val NMSE every {args.val_freq} ep)')

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == 'cuda') else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None:
        logger.info(f'EMA enabled, decay={args.ema_decay}')
    if scaler is not None:
        logger.info('AMP enabled (fp16)')

    # training loop
    best_nmse = 1e9
    best_epoch = -1
    ckpt_path = os.path.join(ckpt_dir,
        f'{args.model}-{args.expansion}X-cr{args.cr}-{args.scenario}-modern')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            args.grad_clip, scaler, ema, epoch, args.epochs,
            step_scheduler_each_batch=step_each_batch)
        logger.info(f'Epoch: {epoch + 1} training loss={train_loss:.3e}')

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            eval_model = ema.module if ema is not None else model
            nmse, rho = evaluate(eval_model, test_loader, device)
            if not step_each_batch:
                # ReduceLROnPlateau: step on the metric we just measured.
                scheduler.step(nmse)
            tag_eval = 'EMA' if ema is not None else 'raw'
            if nmse < best_nmse:
                best_nmse = nmse
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'state_dict': eval_model.state_dict(),
                    'best_nmse': best_nmse,
                    'best_epoch': best_epoch,
                }, ckpt_path)
                logger.info(f'Test Epoch: [{epoch + 1}/{args.epochs}] '
                            f'nmse={nmse:.4f} rho={rho:.4f} ({tag_eval})  '
                            f'[best={best_nmse:.4f} best_epoch={best_epoch + 1}]')
            else:
                logger.info(f'Test Epoch: [{epoch + 1}/{args.epochs}] '
                            f'nmse={nmse:.4f} rho={rho:.4f} ({tag_eval})  '
                            f'(no improve, best={best_nmse:.4f} '
                            f'@ep{best_epoch + 1})')

    logger.info(f'DONE. best_nmse={best_nmse:.4f} @ep{best_epoch + 1}')
    print(f'best_nmse={best_nmse:.4f} @ep{best_epoch + 1}')


if __name__ == '__main__':
    main()
