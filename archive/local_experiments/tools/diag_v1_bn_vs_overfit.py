"""Diagnostic: separate BN drift from overfitting on v1 florence cr=4.

For each cr in {4, 8, 16}:
  1. Load best ckpt, evaluate test in eval mode (standard, uses running stats)
  2. Re-evaluate test in train mode (BN uses batch stats — no running-stat drift)
  3. Re-calibrate BN running stats with train data, eval again
  4. Per-BS test split eval (using bs_id_test) — shows whether the mixture itself is the problem

Interpretation:
  - eval_mode ≈ recal_mode ≈ train_mode  -> pure overfitting
  - train_mode << eval_mode              -> BN drift dominates
  - recal_mode ≈ train_mode << eval_mode -> BN running-stat drift, fixable by recal
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet


def _nmse(pred, gt):
    """Per-sample NMSE on centered (x - 0.5) signal, averaged in dB."""
    s = gt - 0.5
    p = pred - 0.5
    power = (s ** 2).sum(dim=[1, 2, 3])
    err = ((s - p) ** 2).sum(dim=[1, 2, 3])
    return 10 * torch.log10((err / power.clamp_min(1e-12)).mean()).item()


def _eval_loader(model, loader, device):
    losses = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            y = model(x)
            losses.append(_nmse(y, x))
    return float(np.mean(losses))


@torch.no_grad()
def _recal_bn(model, train_loader, device, max_batches=200):
    """Reset BN running stats and re-accumulate from current train batches."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # use exponentially averaged stats — no, use simple avg
            m.num_batches_tracked.zero_()
    model.train()  # BN uses batch stats; also updates running stats
    for i, (x,) in enumerate(train_loader):
        x = x.to(device)
        _ = model(x)
        if i + 1 >= max_batches:
            break
    model.eval()


def diagnose(cr: int, npz_path: str, ckpt_path: str, device: str = "cuda:0"):
    z = np.load(npz_path, allow_pickle=True)
    x_train = torch.from_numpy(z["x_train"])
    x_test = torch.from_numpy(z["x_test"])
    bs_id_test = z["bs_id_test"] if "bs_id_test" in z.files else None
    train_ds = TensorDataset(x_train)
    test_ds = TensorDataset(x_test)
    train_loader = DataLoader(train_ds, batch_size=200, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False, num_workers=0)

    model = dcrnet(reduction=cr, expansion=1).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd["state_dict"], strict=False)

    # 1. eval mode (default, uses running stats)
    model.eval()
    eval_nmse = _eval_loader(model, test_loader, device)

    # 2. train mode (BN uses batch stats — no running-stat usage)
    model.train()
    train_nmse = _eval_loader(model, test_loader, device)

    # 3. eval after recalibrating BN with train data
    _recal_bn(model, train_loader, device, max_batches=200)
    model.eval()
    recal_nmse = _eval_loader(model, test_loader, device)

    # 4. per-BS test (using original eval-mode model, before recal corrupts state)
    # Reload weights so per-BS eval matches step 1's model.
    model.load_state_dict(sd["state_dict"], strict=False)
    model.eval()
    per_bs = {}
    if bs_id_test is not None:
        for bs in np.unique(bs_id_test):
            mask = (bs_id_test == bs)
            x_bs = torch.from_numpy(z["x_test"][mask])
            loader_bs = DataLoader(TensorDataset(x_bs), batch_size=200, shuffle=False)
            per_bs[int(bs)] = _eval_loader(model, loader_bs, device)

    return {
        "cr": cr,
        "eval_NMSE": eval_nmse,
        "train_mode_NMSE": train_nmse,
        "recal_NMSE": recal_nmse,
        "per_bs_NMSE": per_bs,
    }


def main():
    npz = "/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_florence_3GHz_32x1024.npz"
    for cr in (4, 8, 16):
        ckpt = f"/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints/v1-1X-cr{cr}-in-florence_v1_cr{cr}"
        if not os.path.isfile(ckpt):
            print(f"cr={cr}: ckpt missing, skip")
            continue
        r = diagnose(cr, npz, ckpt)
        print(f"\n=== v1 florence cr={cr} ===")
        print(f"  eval mode (running stats):  {r['eval_NMSE']:+.3f} dB")
        print(f"  train mode (batch stats):   {r['train_mode_NMSE']:+.3f} dB")
        print(f"  eval after BN recal:        {r['recal_NMSE']:+.3f} dB")
        gap_bn = r['eval_NMSE'] - r['train_mode_NMSE']
        gap_recal = r['eval_NMSE'] - r['recal_NMSE']
        print(f"  eval - train (BN drift):    {gap_bn:+.3f} dB  (positive = worse via running stats)")
        print(f"  eval - recal (drift fixable): {gap_recal:+.3f} dB")
        if r['per_bs_NMSE']:
            per_bs = ", ".join(f"BS{k}={v:.2f}" for k, v in sorted(r['per_bs_NMSE'].items()))
            mean = np.mean(list(r['per_bs_NMSE'].values()))
            spread = max(r['per_bs_NMSE'].values()) - min(r['per_bs_NMSE'].values())
            print(f"  per-BS:  {per_bs}")
            print(f"  per-BS mean={mean:+.3f}  spread={spread:.3f}")


if __name__ == "__main__":
    main()
