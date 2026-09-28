"""Compare Mix5-trained DCRNetV2-unified weights with the paper's scene-wise NMSE table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rtcsi import load_rtcsi_split
from train import build_model, evaluate


PAPER_NMSE = {
    "etoile": (-20.63, -16.89, -14.10),  # Paris / Place de l'Étoile
    "florence": (-24.40, -19.50, -15.86),
    "munich": (-20.16, -15.90, -12.67),
    "ZJU": (-17.82, -14.62, -12.24),
    "SUTD": (-17.53, -14.61, -12.28),
}
CRS = (4, 8, 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="RT-CSI release data directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tolerance", type=float, default=0.13, help="Maximum allowed difference from the paper, in dB")
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    weights = Path(__file__).resolve().parents[1] / "weights" / "rtcsi-mix5" / "dcrnetv2-unified"
    models = {}
    for cr in CRS:
        checkpoint = torch.load(weights / f"cr{cr}.pt", map_location="cpu", weights_only=True)
        model = build_model("unified", cr)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[cr] = model.to(device).eval()

    failures = 0
    print("scene      CR  paper_dB  measured_dB  delta_dB  result", flush=True)
    for scene, expected_values in PAPER_NMSE.items():
        path = args.data / f"csi_{scene}_3GHz_32x1024.npz"
        test = load_rtcsi_split(path, "test")
        loader = DataLoader(TensorDataset(test), batch_size=200, shuffle=False)
        for cr, expected in zip(CRS, expected_values):
            measured, _ = evaluate(loader, models[cr], device, metric="paper")
            delta = measured - expected
            matches = abs(delta) <= args.tolerance
            failures += not matches
            print(
                f"{scene:9} {cr:2}  {expected:8.2f}  {measured:11.4f}  "
                f"{delta:+8.4f}  {'PASS' if matches else 'FAIL'}",
                flush=True,
            )
    print(f"Within {args.tolerance:.2f} dB of the paper: {len(PAPER_NMSE) * len(CRS) - failures}/{len(PAPER_NMSE) * len(CRS)} values.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
