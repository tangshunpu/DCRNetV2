"""Inspect COST2100 or RT-CSI split shapes before training/evaluation."""

from __future__ import annotations

import argparse

from dataset.cost2100 import Cost2100DataLoader
from dataset.rtcsi import RTCSIDataLoader


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cost2100", "rtcsi"], default="cost2100")
    parser.add_argument("--data", help="COST2100 directory or RT-CSI train/val NPZ")
    parser.add_argument("--test-data", help="Separate RT-CSI test NPZ")
    parser.add_argument("--scenario", choices=["in", "out"], help="COST2100 indoor/outdoor scenario (default: in)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative")
    if args.dataset == "cost2100" and args.test_data:
        parser.error("--test-data is only used with --dataset rtcsi")
    if args.dataset == "rtcsi" and not args.data:
        parser.error("--data must point to an RT-CSI NPZ")
    if args.dataset == "rtcsi" and args.scenario:
        parser.error("--scenario is only used with --dataset cost2100")

    if args.dataset == "cost2100":
        loader = Cost2100DataLoader(
            args.data or "./data/COST2100", scenario=args.scenario or "in",
            batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=False, prefetch=False,
        )
    else:
        loader = RTCSIDataLoader(
            args.data, test_path=args.test_data,
            batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=False, prefetch=False,
        )
    for name, dataset in (("train", loader.train_dataset), ("val", loader.val_dataset), ("test", loader.test_dataset)):
        tensor = dataset.tensors[0]
        print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} min={tensor.min():.4g} max={tensor.max():.4g}")
    train_loader, _, test_loader = loader()
    print(f"loader train batch: {tuple(next(iter(train_loader))[0].shape)}")
    print(f"loader test batch: {tuple(next(iter(test_loader))[0].shape)}")


if __name__ == "__main__":
    main()
