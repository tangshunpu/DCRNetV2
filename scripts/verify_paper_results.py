"""Check curated DCRNetV2 checkpoints against the paper's COST2100 NMSE table.

Each scenario's test tensors are loaded once. The published metric averages
per-batch NMSE in dB with batch size 200, so keep that setting when comparing
to the reported two-decimal values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cost2100 import load_cost2100_test_tensors
from train import build_model, evaluate


PAPER_NMSE = {
    "mini": {"in": [-31.00, -19.23, -14.18, -9.55], "out": [-10.83, -7.20, -4.83, -3.14]},
    "small": {"in": [-31.12, -20.51, -14.64, -9.16], "out": [-11.44, -7.70, -5.17, -3.31]},
    "base": {"in": [-32.75, -20.07, -14.73, -9.35], "out": [-11.89, -7.83, -5.51, -3.56]},
    "unified": {"in": [-32.84, -20.36, -14.62, -9.69], "out": [-12.99, -8.65, -5.85, -3.79]},
}
CRS = (4, 8, 16, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="./data/COST2100")
    parser.add_argument("--scenario", choices=["in", "out", "both"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tolerance", type=float, default=0.0051, help="dB tolerance for two-decimal paper values")
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    scenarios = ("in", "out") if args.scenario == "both" else (args.scenario,)
    failures = 0
    count = 0

    print("variant  scenario  CR  paper_dB  measured_dB  delta_dB  result", flush=True)
    for scenario in scenarios:
        test, raw_test = load_cost2100_test_tensors(args.data, scenario)
        loader = DataLoader(TensorDataset(test, raw_test), batch_size=200, shuffle=False, num_workers=0)
        for variant, by_scenario in PAPER_NMSE.items():
            for cr, expected in zip(CRS, by_scenario[scenario]):
                path = Path(__file__).resolve().parents[1] / "weights" / f"dcrnetv2-{variant}" / scenario / f"cr{cr}.pt"
                checkpoint = torch.load(path, map_location="cpu")
                model = build_model(variant, cr)
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device).eval()
                measured, _ = evaluate(loader, model, device, metric="paper")
                delta = measured - expected
                matches = abs(delta) <= args.tolerance
                failures += not matches
                count += 1
                print(
                    f"{variant:8} {scenario:8} {cr:2}  {expected:8.2f}  {measured:11.4f}  "
                    f"{delta:+8.4f}  {'PASS' if matches else 'FAIL'}",
                    flush=True,
                )
                del model
        del loader, test, raw_test

    print(f"Matched {count - failures}/{count} paper NMSE values (tolerance {args.tolerance:.4f} dB).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
