"""Evaluate a curated or local DCRNetV2/LRP checkpoint on COST2100 test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.cost2100 import load_cost2100_test_tensors
from train import VARIANTS, build_model, evaluate


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCRNetV2/LRP on COST2100 test data")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--scenario", required=True, choices=["in", "out"])
    parser.add_argument("--cr", required=True, type=int, choices=[4, 8, 16, 32])
    parser.add_argument("--data", default="./data/COST2100", help="COST2100 dataset root")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to weights/<variant>/<scenario>/cr<cr>.pt")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["paper", "global"], default="paper", help="Paper averages per-batch dB at batch size 200")
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    name = args.variant if args.variant.startswith("lrp-") else f"dcrnetv2-{args.variant}"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("weights") / name / args.scenario / f"cr{args.cr}.pt"
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        parser.error(f"Expected a checkpoint dictionary in {checkpoint_path}")
    for key, expected in (("variant", name), ("scenario", args.scenario), ("cr", args.cr)):
        if key in checkpoint and checkpoint[key] != expected:
            parser.error(f"Checkpoint {key}={checkpoint[key]!r} does not match requested {expected!r}")

    model = build_model(args.variant, args.cr)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device).eval()
    test, raw_test = load_cost2100_test_tensors(args.data, args.scenario)
    loader = DataLoader(
        TensorDataset(test, raw_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    nmse, rho = evaluate(loader, model, device, metric=args.metric)
    print(f"checkpoint: {checkpoint_path}")
    print(f"scenario={args.scenario} cr={args.cr} samples={len(test)} device={device} metric={args.metric} batch_size={args.batch_size}")
    print(f"NMSE={nmse:.4f} dB  rho={rho:.4f}")


if __name__ == "__main__":
    main()
