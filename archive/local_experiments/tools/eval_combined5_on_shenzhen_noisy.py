"""Robustness sweep: combined5 models on SHENZHEN_test with AWGN injection.

For each (model, cr, sigma):
  x_clean -> add AWGN(sigma) -> model -> y_pred
  NMSE = || x_clean - y_pred ||^2 / || x_clean ||^2  (vs CLEAN ground truth)

This is the channel-estimation-error robustness test from the CSI feedback
literature (CsiNet, TransNet, etc.). sigma is the per-pixel AWGN std on
the centered (x - 0.5) tensor; the dataset is per-sample max-abs normalised
so peak |signal| ~ 0.5, so sigma=0.05 ≈ SNR_peak 20 dB, sigma=0.1 ≈ 14 dB.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet, crnet, clnet


def _v9u(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


def _make_with_env(env_changes, factory, cr):
    old = {k: os.environ.get(k) for k in env_changes}
    try:
        for k, v in env_changes.items():
            os.environ[k] = v
        m = factory(cr)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return m


MODELS = [
    ("clnet",     "clnet",    lambda cr: clnet(reduction=cr, expansion=1),    "combined5_clnet_cr"),
    ("crnet",     "crnet",    lambda cr: crnet(reduction=cr, expansion=1),    "combined5_crnet_cr"),
    ("v1",        "v1",       lambda cr: dcrnet(reduction=cr, expansion=1),   "combined5_v1_cr"),
    ("v5",        "v5",       lambda cr: dcrnet_v5(reduction=cr, expansion=1,
                                                     ranks=16, r_enc=512),    "combined5_v5_cr"),
    ("v9u",       "v9",       _v9u,                                            "combined5_v9u_cr"),
    ("v9u_sase",  "v9",       lambda cr: _make_with_env(
                                {"DCRNET_V9_SA": "1", "DCRNET_V9_SE": "1"},
                                _v9u, cr),                                     "combined5_v9u_sase_cr"),
    ("v9u_warm",  "v9",       _v9u,                                            "combined5_v9u_warm_cr"),
    ("transnet",  "transnet", lambda cr: transnet(reduction=cr, expansion=1), "combined5_transnet_cr"),
]


def _nmse_vs_clean(pred, clean):
    """NMSE in dB, compared to CLEAN ground truth (channel estimation error
    robustness). Both pred and clean in [0, 1]; centred at 0.5 for NMSE."""
    s = clean - 0.5
    p = pred  - 0.5
    power = (s ** 2).sum(dim=[1, 2, 3])
    err = ((s - p) ** 2).sum(dim=[1, 2, 3])
    return 10 * torch.log10((err / power.clamp_min(1e-12)).mean()).item()


@torch.no_grad()
def _eval_noisy(model, loader, device, sigma, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for (x_clean,) in loader:
        x_clean = x_clean.to(device)
        if sigma > 0:
            noise = torch.randn(x_clean.shape, generator=g, device=device) * sigma
            x_noisy = (x_clean + noise).clamp(0.0, 1.0)
        else:
            x_noisy = x_clean
        y = model(x_noisy)
        losses.append(_nmse_vs_clean(y, x_clean))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.1],
                    help="AWGN std values to test (on centred tensor)")
    ap.add_argument("--test-npz",
                    default="/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_SHENZHEN_test_3GHz_32x1024.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sz = np.load(args.test_npz, allow_pickle=True)
    x_test = torch.from_numpy(sz["x_test"])
    print(f"test set: {os.path.basename(args.test_npz)}  N={len(x_test)}")
    loader = DataLoader(TensorDataset(x_test), batch_size=200, shuffle=False)

    ckpt_dir = "/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints"
    # rows[model] = {(cr, sigma): nmse}
    rows = {}

    for mkey, mpfx, factory, tag_pfx in MODELS:
        for cr in (4, 8, 16):
            ckpt = f"{ckpt_dir}/{mpfx}-1X-cr{cr}-in-{tag_pfx}{cr}"
            if not os.path.isfile(ckpt):
                print(f"{mkey:10s} cr{cr:>2d}  MISSING")
                continue
            sd = torch.load(ckpt, map_location=device,
                            weights_only=False)["state_dict"]
            model = factory(cr).to(device)
            model.load_state_dict(sd, strict=False)
            model.eval()
            for sigma in args.sigmas:
                n = _eval_noisy(model, loader, device, sigma, seed=args.seed)
                rows.setdefault(mkey, {})[(cr, sigma)] = n
                print(f"{mkey:10s} cr{cr:>2d}  sigma={sigma:.3f}  NMSE={n:+.3f} dB")
            del model
            torch.cuda.empty_cache()

    # Final table per cr
    print()
    sigmas = args.sigmas
    for cr in (4, 8, 16):
        print(f"\n=== cr = {cr} =================================")
        header = f"{'model':10s}  " + "  ".join(f"σ={s:.3f}" for s in sigmas)
        print(header)
        print("-" * len(header))
        for mkey, *_ in MODELS:
            cells = []
            for s in sigmas:
                v = rows.get(mkey, {}).get((cr, s))
                cells.append(f"{v:+7.3f}" if v is not None else "    --")
            print(f"{mkey:10s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
