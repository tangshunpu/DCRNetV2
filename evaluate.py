"""Evaluate a DCRNetV2/LRP checkpoint on COST2100 or RT-CSI test data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.cost2100 import load_cost2100_test_tensors
from dataset.rtcsi import load_rtcsi_split
from train import VARIANTS, build_model, evaluate


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCRNetV2/LRP on COST2100 or RT-CSI test data")
    parser.add_argument("--dataset", choices=["cost2100", "rtcsi"], default="cost2100")
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--scenario", choices=["in", "out"], help="COST2100 indoor/outdoor scenario")
    parser.add_argument("--cr", required=True, type=int, choices=[4, 8, 16, 32])
    parser.add_argument("--data", default="./data/COST2100", help="COST2100 directory or RT-CSI test NPZ")
    parser.add_argument("--checkpoint", help="Checkpoint path; defaults to the curated COST2100 or RT-CSI weight")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metric", choices=["paper", "global"], default="paper", help="Paper averages per-batch dB at batch size 200")
    args = parser.parse_args(argv)

    if args.dataset == "cost2100" and args.scenario is None:
        parser.error("--scenario is required for COST2100")
    if args.dataset == "rtcsi" and args.scenario:
        parser.error("--scenario is only used with --dataset cost2100")
    if args.dataset == "rtcsi" and Path(args.data).suffix.lower() != ".npz":
        parser.error("--data must point to an RT-CSI .npz file")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    name = args.variant if args.variant.startswith("lrp-") else f"dcrnetv2-{args.variant}"
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.dataset == "rtcsi":
        checkpoint_path = Path("weights") / "rtcsi-mix5" / name / f"cr{args.cr}.pt"
    else:
        checkpoint_path = Path("weights") / name / args.scenario / f"cr{args.cr}.pt"
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        parser.error(f"Expected a checkpoint dictionary in {checkpoint_path}")
    expected_metadata = [("variant", name), ("cr", args.cr)]
    if args.dataset == "cost2100":
        expected_metadata.append(("scenario", args.scenario))
    for key, expected in expected_metadata:
        if key in checkpoint and checkpoint[key] != expected:
            parser.error(f"Checkpoint {key}={checkpoint[key]!r} does not match requested {expected!r}")

    model = build_model(args.variant, args.cr)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    if args.dataset == "cost2100":
        test, raw_test = load_cost2100_test_tensors(args.data, args.scenario)
        test_dataset = TensorDataset(test, raw_test)
    else:
        test = load_rtcsi_split(args.data, "test")
        test_dataset = TensorDataset(test)
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    nmse, rho = evaluate(loader, model, device, metric=args.metric)
    print(f"checkpoint: {checkpoint_path}")
    if "training_data" in checkpoint:
        print(f"checkpoint trained on: {checkpoint['training_data']}")
    scenario_text = f" scenario={args.scenario}" if args.scenario else ""
    print(f"dataset={args.dataset}{scenario_text} cr={args.cr} samples={len(test)} device={device} metric={args.metric} batch_size={args.batch_size}")
    metrics = f"NMSE={nmse:.4f} dB"
    if rho is not None:
        metrics += f"  rho={rho:.4f}"
    print(metrics)


if __name__ == "__main__":
    main()
