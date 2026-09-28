"""Train DCRNetV2 or LRP on COST2100 or normalized RT-CSI NPZ data.

Examples:
    python train.py --variant base --scenario out --cr 4 --data ./data/COST2100
    python train.py --variant lrp-r16-d512 --scenario out --cr 4 --data ./data/COST2100
    python train.py --variant mini --scenario in --cr 16 --epochs 1 --workers 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset.cost2100 import Cost2100DataLoader
from dataset.rtcsi import RTCSIDataLoader
from models.dcrnetv2 import (
    dcrnetv2_base,
    dcrnetv2_large_in4,
    dcrnetv2_large_out,
    dcrnetv2_mini,
    dcrnetv2_small,
    dcrnetv2_unified,
)
from models.lrp import lrp_r4_d128, lrp_r4_d512, lrp_r8_d512, lrp_r16_d512
from utils import AverageMeter, WarmUpCosineAnnealingLR, evaluator, nmse_db, nmse_ratio


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


def convert_legacy_dcrnetv2_state(model: torch.nn.Module, state_dict: dict) -> dict:
    """Adapt early DCRNetV2 checkpoints with fixed-slope encoder activations.

    Early experiments used LeakyReLU(slope=0.3) in the DCR encoder, where the
    public model uses PReLU. Setting those PReLU weights to 0.3 is equivalent
    at inference. Old ``conv2.0`` names and thop counters are also normalized.
    The caller must still use strict loading to catch other incompatibilities.
    """

    converted = {}
    for key, value in state_dict.items():
        if key.split(".")[-1] in {"total_ops", "total_params"}:
            continue
        new_key = key.replace("enc_dilate.block.conv2.0.", "enc_dilate.block.conv2.")
        if new_key in converted:
            raise ValueError(f"Duplicate checkpoint key after conversion: {new_key}")
        converted[new_key] = value
    expected = model.state_dict()
    fixed_slope_keys = [f"enc_dilate.block.conv1.{i}.weight" for i in (1, 3, 5, 7, 9)]
    fixed_slope_keys += ["enc_dilate.block.prelu1.weight", "enc_dilate.block.prelu2.weight"]
    for key in fixed_slope_keys:
        if key in expected and key not in converted:
            converted[key] = torch.full_like(expected[key], 0.3)
    return converted


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
def evaluate(loader, model, device, metric="paper"):
    """Compute paper-style batch NMSE or global NMSE; rho needs raw CSI."""

    if metric not in {"paper", "global"}:
        raise ValueError(f"Unknown NMSE metric: {metric}")
    nmse_meter = AverageMeter("nmse_db")
    ratio_sum = 0.0
    rho_meter = AverageMeter("rho")
    model.eval()
    for batch in loader:
        sparse_gt = batch[0].to(device, non_blocking=True)
        sparse_pred = model(sparse_gt)
        bs = sparse_gt.size(0)
        if len(batch) > 1:
            raw_gt = batch[1].to(device, non_blocking=True)
            rho, batch_nmse = evaluator(sparse_pred, sparse_gt, raw_gt)
            rho_meter.update(rho, n=bs)
        else:
            batch_nmse = nmse_db(sparse_pred, sparse_gt)
        nmse_meter.update(batch_nmse, n=bs)
        if metric == "global":
            ratio_sum += nmse_ratio(sparse_pred, sparse_gt).sum().item()
    if not nmse_meter.count:
        raise ValueError("Cannot evaluate an empty dataset")
    nmse = nmse_meter.avg if metric == "paper" else 10 * math.log10(ratio_sum / nmse_meter.count)
    return nmse, (rho_meter.avg if rho_meter.count else None)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "dataset": args.dataset,
            "variant": args.variant if args.variant.startswith("lrp-") else f"dcrnetv2-{args.variant}",
            "cr": args.cr,
            "scenario": args.scenario,
            **metrics,
            "args": vars(args),
        },
        path,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train DCRNetV2/LRP on COST2100 or RT-CSI")
    parser.add_argument("--dataset", choices=["cost2100", "rtcsi"], default="cost2100")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS), help="Model variant; LRP variants use lrp-r{r}-d{d}")
    parser.add_argument("--scenario", choices=["in", "out"], help="COST2100 indoor/outdoor scenario")
    parser.add_argument("--cr", type=int, required=True, choices=[4, 8, 16, 32], help="Compression ratio")
    parser.add_argument("--data", default="./data/COST2100", help="COST2100 directory or RT-CSI train/val NPZ")
    parser.add_argument("--test-data", help="RT-CSI test NPZ if separate from train/val NPZ")
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
    parser.add_argument("--legacy-checkpoint", action="store_true", help="Convert early DCRNetV2 weights; requires --finetune-from")
    parser.add_argument("--gate-reset", type=float, default=None, help="Reset hybrid rank_gate.bias after loading weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-prefetch", action="store_true", help="Disable CUDA prefetcher")
    args = parser.parse_args(argv)

    if args.dataset == "cost2100" and args.scenario is None:
        parser.error("--scenario is required for COST2100")
    if args.dataset == "cost2100" and args.test_data:
        parser.error("--test-data is only used with --dataset rtcsi")
    if args.dataset == "rtcsi" and args.scenario:
        parser.error("--scenario is only used with --dataset cost2100")
    if args.dataset == "rtcsi" and Path(args.data).suffix.lower() != ".npz":
        parser.error("--data must point to an RT-CSI .npz file")
    if args.legacy_checkpoint and (not args.finetune_from or args.resume or args.variant.startswith("lrp-")):
        parser.error("--legacy-checkpoint requires --finetune-from and a DCRNetV2 variant")

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

    loader_class = Cost2100DataLoader if args.dataset == "cost2100" else RTCSIDataLoader
    data_kwargs = {"scenario": args.scenario} if args.dataset == "cost2100" else {"test_path": args.test_data}
    train_loader, val_loader, test_loader = loader_class(
        args.data, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=pin_memory, prefetch=not args.no_prefetch, **data_kwargs,
    )()
    scheduler = WarmUpCosineAnnealingLR(
        optimizer,
        T_max=max(1, len(train_loader) * args.epochs),
        T_warmup=args.warmup_epochs * len(train_loader),
        eta_min=args.eta_min,
    )

    start_epoch = 0
    best_val_nmse = float("inf")
    best_epoch = -1

    load_path = args.resume or args.finetune_from
    if load_path:
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if args.legacy_checkpoint:
            model.load_state_dict(convert_legacy_dcrnetv2_state(model, state_dict), strict=True)
        else:
            model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {load_path}")
        if args.resume:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            if checkpoint.get("selection_split") == "val":
                best_val_nmse = float(checkpoint.get("best_val_nmse", best_val_nmse))
                best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        if args.gate_reset is not None:
            reset_gate_bias(model, args.gate_reset)
            print(f"Reset rank_gate.bias to {args.gate_reset}")

    n_params = sum(p.numel() for p in model.parameters())
    model_label = args.variant.replace("lrp", "LRP", 1) if args.variant.startswith("lrp-") else f"DCRNetV2-{args.variant}"
    data_label = args.scenario if args.dataset == "cost2100" else f"rtcsi-{Path(args.data).stem}"
    run_name = f"{model_label}-{data_label}-cr{args.cr}"
    ckpt_best = ckpt_dir / f"{run_name}-best.pt"
    ckpt_last = ckpt_dir / f"{run_name}-last.pt"
    log_path = log_dir / f"{run_name}.jsonl"

    print(f"Model: {run_name}  params={n_params/1e3:.1f}K  device={device}")
    print(f"Data: {args.data}  train_batches={len(train_loader)}  val_batches={len(val_loader)}  test_batches={len(test_loader)}")
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
            val_nmse, _ = evaluate(val_loader, model, device)
            improved = val_nmse < best_val_nmse
            if improved:
                best_val_nmse = val_nmse
                best_epoch = epoch
            row.update({"val_nmse": val_nmse, "best_val_nmse": best_val_nmse, "best_epoch": best_epoch + 1})
            tag = " NEW BEST" if improved else ""
            print(
                f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.3e}  "
                f"val_NMSE={val_nmse:.4f} dB  "
                f"best={best_val_nmse:.4f} @ep{best_epoch+1}{tag}  ({row['elapsed_sec']:.1f}s)"
            )
            if improved:
                save_checkpoint(
                    ckpt_best,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"val_nmse": val_nmse, "best_val_nmse": best_val_nmse, "best_epoch": best_epoch, "selection_split": "val"},
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
            {"best_val_nmse": best_val_nmse, "best_epoch": best_epoch, "selection_split": "val"},
            args,
        )

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was produced; check --epochs and --val-freq")
    best_state = torch.load(ckpt_best, map_location=device)
    model.load_state_dict(best_state["state_dict"])
    test_nmse, test_rho = evaluate(test_loader, model, device)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "test", "checkpoint": str(ckpt_best), "nmse": test_nmse, "rho": test_rho}) + "\n")
    print(f"\n=> Best validation NMSE: {best_val_nmse:.4f} dB @ epoch {best_epoch+1}")
    test_metrics = f"NMSE={test_nmse:.4f} dB"
    if test_rho is not None:
        test_metrics += f"  rho={test_rho:.4f}"
    print(f"=> Held-out test: {test_metrics}")
    print(f"=> Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
