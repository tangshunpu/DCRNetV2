"""Few-shot fine-tuning sweep on Shenzhen for combined5 cr=4 ckpts.

For each (model, train_size):
  1. Load combined5-trained cr=4 ckpt (8 models)
  2. Fine-tune on `csi_SHENZHEN_train{N}_3GHz_32x1024.npz` for fixed epochs
  3. Evaluate NMSE on the existing SHENZHEN_test set (40k samples)
  4. Also record the "zero-shot" NMSE (no fine-tuning)

Produces a (model, train_size) table of NMSE in dB.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
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
    ("transnet",  "transnet", lambda cr: transnet(reduction=cr, expansion=1), "combined5_transnet_cr"),
]


def _nmse(pred, gt):
    s = gt - 0.5
    p = pred - 0.5
    power = (s ** 2).sum(dim=[1, 2, 3])
    err = ((s - p) ** 2).sum(dim=[1, 2, 3])
    return 10 * torch.log10((err / power.clamp_min(1e-12)).mean()).item()


@torch.no_grad()
def _eval(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        x = batch[0].to(device)
        y = model(x)
        losses.append(_nmse(y, x))
    return float(np.mean(losses))


def fine_tune(model, train_loader, val_loader, device,
              lr=1e-4, epochs=200, patience=50, log_prefix=""):
    """Fine-tune with Adam + early stopping on val NMSE."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_val_nmse = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y = model(x)
            loss = crit(y, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = _eval(model, val_loader, device)
        if v < best_val_nmse - 0.02:
            best_val_nmse = v
            best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return best_val_nmse, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[200, 400, 800, 1600, 3200])
    ap.add_argument("--from-scratch", action="store_true",
                    help="skip pretrained ckpt; init random (no transfer learning)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-pfx",
                    default="/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_SHENZHEN_train")
    ap.add_argument("--test-npz",
                    default="/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_SHENZHEN_test_3GHz_32x1024.npz")
    a = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sz = np.load(a.test_npz, allow_pickle=True)
    x_test = torch.from_numpy(sz["x_test"])
    test_loader = DataLoader(TensorDataset(x_test), batch_size=200, shuffle=False)
    print(f"test set: N={len(x_test)}")

    ckpt_dir = "/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints"

    # results[model][N] = NMSE; results[model]["zero_shot"] = NMSE without ft
    results: dict[str, dict] = {}

    for mkey, mpfx, factory, tag_pfx in MODELS:
        results[mkey] = {}
        if a.from_scratch:
            sd_init = None
            zs_str = "(skipped, from-scratch mode)"
            results[mkey]["zs"] = float("nan")
        else:
            ckpt = f"{ckpt_dir}/{mpfx}-1X-cr{a.cr}-in-{tag_pfx}{a.cr}"
            if not os.path.isfile(ckpt):
                print(f"{mkey:10s}  MISSING {ckpt}")
                continue
            sd_init = torch.load(ckpt, map_location=device,
                                  weights_only=False)["state_dict"]
            # Zero-shot NMSE (no fine-tuning)
            model = factory(a.cr).to(device)
            model.load_state_dict(sd_init, strict=False)
            zs = _eval(model, test_loader, device)
            results[mkey]["zs"] = zs
            zs_str = f"NMSE = {zs:+.3f} dB"
            del model
            torch.cuda.empty_cache()
        print(f"{mkey:10s}  zero-shot   {zs_str}")

        # Fine-tune at each train size
        for N in a.sizes:
            tr = np.load(f"{a.train_pfx}{N}_3GHz_32x1024.npz", allow_pickle=True)
            x_tr = torch.from_numpy(tr["x_train"])
            x_va = torch.from_numpy(tr["x_val"])
            tr_loader = DataLoader(TensorDataset(x_tr), batch_size=a.batch_size, shuffle=True)
            va_loader = DataLoader(TensorDataset(x_va), batch_size=200, shuffle=False)

            # Fresh init (or load pretrained weights)
            model = factory(a.cr).to(device)
            if sd_init is not None:
                model.load_state_dict(sd_init, strict=False)

            t0 = time.time()
            best_val, ep_used = fine_tune(model, tr_loader, va_loader, device,
                                            lr=a.lr, epochs=a.epochs,
                                            patience=a.patience,
                                            log_prefix=f"{mkey} N={N}")
            ft_nmse = _eval(model, test_loader, device)
            results[mkey][N] = ft_nmse
            dt = time.time() - t0
            print(f"{mkey:10s}  N={N:>4d}  ep={ep_used:>3d}  "
                  f"val_best={best_val:+7.3f}  test_NMSE={ft_nmse:+7.3f} dB  ({dt:.0f}s)")
            del model
            torch.cuda.empty_cache()

    # Final table
    print()
    print("=" * 80)
    sizes = a.sizes
    header = f"{'model':10s}  {'zero-shot':>10s}  " + "  ".join(f"N={N}" for N in sizes)
    print(header)
    print("-" * len(header))
    for mkey, *_ in MODELS:
        r = results.get(mkey, {})
        cells = [f"{r.get('zs', float('nan')):+7.3f}"]
        for N in sizes:
            v = r.get(N)
            cells.append(f"{v:+7.3f}" if v is not None else "    --")
        print(f"{mkey:10s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
