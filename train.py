"""Train DCRNetV2 or LRP on COST2100.

Examples:
    python train.py --variant base --scenario out --cr 4 --data ./data/COST2100
    python train.py --variant lrp-r16-d512 --scenario out --cr 4 --data ./data/COST2100
    python train.py --variant mini --scenario in --cr 16 --epochs 1 --workers 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset.cost2100 import Cost2100DataLoader
from models.dcrnetv2 import (
    dcrnetv2_base,
    dcrnetv2_large_in4,
    dcrnetv2_large_out,
    dcrnetv2_mini,
    dcrnetv2_small,
    dcrnetv2_unified,
)
from models.lrp import lrp_r4_d128, lrp_r4_d512, lrp_r8_d512, lrp_r16_d512
from utils import AverageMeter, WarmUpCosineAnnealingLR, evaluator


VARIANTS = {
    "mini": dcrnetv2_mini,
    "small": dcrnetv2_small,
    "base": dcrnetv2_base,
    "unified": dcrnetv2_unified,
    "large-out": dcrnetv2_large_out,
    "large-in4": dcrnetv2_large_in4,
    # LRP low-rank-prior variants; suffix is decoder rank r and encoder dim d.
    "lrp-r4-d128": lrp_r4_d128,
    "lrp-r4-d512": lrp_r4_d512,
    "lrp-r8-d512": lrp_r8_d512,
    "lrp-r16-d512": lrp_r16_d512,
}


def build_model(variant: str, reduction: int) -> torch.nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[variant](reduction=reduction)


def reset_gate_bias(model: torch.nn.Module, value: float) -> None:
    """Reset hybrid decoder gate bias for fine-tuning recipes."""

    if not hasattr(model.decoder, "extra"):
        raise RuntimeError("--gate-reset requires a hybrid decoder variant")
    with torch.no_grad():
        model.decoder.extra.rank_gate.bias.fill_(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(loader, model, criterion, optimizer, scheduler, device):
    meter = AverageMeter("loss")
    model.train()
    for batch in loader:
        sparse_gt = batch[0].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        sparse_pred = model(sparse_gt)
        loss = criterion(sparse_pred, sparse_gt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        meter.update(loss.item(), n=sparse_gt.size(0))
    return meter.avg


@torch.no_grad()
def evaluate(loader, model, device):
    nmse_meter = AverageMeter("nmse")
    rho_meter = AverageMeter("rho")
    model.eval()
    for batch in loader:
        sparse_gt = batch[0].to(device, non_blocking=True)
        raw_gt = batch[1].to(device, non_blocking=True)
        sparse_pred = model(sparse_gt)
        rho, nmse = evaluator(sparse_pred, sparse_gt, raw_gt)
        bs = sparse_gt.size(0)
        nmse_meter.update(nmse, n=bs)
        rho_meter.update(rho, n=bs)
    return nmse_meter.avg, rho_meter.avg


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            **metrics,
            "args": vars(args),
        },
        path,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train DCRNetV2/LRP on COST2100")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS), help="Model variant; LRP variants use lrp-r{r}-d{d}")
    parser.add_argument("--scenario", required=True, choices=["in", "out"], help="COST2100 indoor/outdoor scenario")
    parser.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32], help="Compression ratio")
    parser.add_argument("--data", default="./data/COST2100", help="COST2100 dataset root")
    parser.add_argument("--outputs", default="./outputs", help="Output dir for checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eta-min", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index; ignored when CUDA is unavailable")
    parser.add_argument("--val-freq", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--resume", default=None, help="Resume full training state from checkpoint")
    parser.add_argument("--finetune-from", default=None, help="Load model weights from checkpoint")
    parser.add_argument("--gate-reset", type=float, default=None, help="Reset hybrid rank_gate.bias after loading weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prefetch", action="store_true", help="Disable CUDA prefetcher")
    args = parser.parse_args(argv)

    if args.seed is not None:
        seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
        pin_memory = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    outputs = Path(args.outputs)
    ckpt_dir = outputs / "checkpoints"
    log_dir = outputs / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.variant, args.cr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loader, _, test_loader = Cost2100DataLoader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario,
        prefetch=not args.no_prefetch,
    )()
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) * args.epochs),
        T_warmup=args.warmup_epochs * len(train_loader),
        eta_min=args.eta_min,
    )

    start_epoch = 0
    best_nmse = float("inf")
    best_epoch = -1

    load_path = args.resume or args.finetune_from
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {load_path}")
        if args.resume:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_nmse = float(checkpoint.get("best_nmse", checkpoint.get("nmse", best_nmse)))
            best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", best_epoch)))
        if args.gate_reset is not None:
            reset_gate_bias(model, args.gate_reset)
            print(f"Reset rank_gate.bias to {args.gate_reset}")

    n_params = sum(p.numel() for p in model.parameters())
    model_label = args.variant.replace("lrp", "LRP", 1) if args.variant.startswith("lrp-") else f"DCRNetV2-{args.variant}"
    run_name = f"{model_label}-{args.scenario}-cr{args.cr}"
    ckpt_best = ckpt_dir / f"{run_name}-best.pt"
    ckpt_last = ckpt_dir / f"{run_name}-last.pt"
    log_path = log_dir / f"{run_name}.jsonl"

    print(f"Model: {run_name}  params={n_params/1e3:.1f}K  device={device}")
    print(f"Data: {args.data}  train_batches={len(train_loader)}  test_batches={len(test_loader)}")
    print(f"Logs: {log_path}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, device)
        should_eval = (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - t0,
        }
        if should_eval:
            nmse, rho = evaluate(test_loader, model, device)
            improved = nmse < best_nmse
            if improved:
                best_nmse = nmse
                best_epoch = epoch
            row.update({"nmse": nmse, "rho": rho, "best_nmse": best_nmse, "best_epoch": best_epoch + 1})
            tag = " NEW BEST" if improved else ""
            print(
                f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.3e}  "
                f"NMSE={nmse:.4f} dB  rho={rho:.4f}  "
                f"best={best_nmse:.4f} @ep{best_epoch+1}{tag}  ({row['elapsed_sec']:.1f}s)"
            )
            if improved:
                save_checkpoint(
                    ckpt_best,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"nmse": nmse, "rho": rho, "best_nmse": best_nmse, "best_epoch": best_epoch},
                    args,
                )
        else:
            print(f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.3e}  ({row['elapsed_sec']:.1f}s)")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        save_checkpoint(
            ckpt_last,
            model,
            optimizer,
            scheduler,
            epoch,
            {"best_nmse": best_nmse, "best_epoch": best_epoch},
            args,
        )

    print(f"\n=> Final best NMSE: {best_nmse:.4f} dB @ epoch {best_epoch+1}")
    print(f"=> Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
