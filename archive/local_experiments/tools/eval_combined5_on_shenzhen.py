"""Cross-domain test: load v1/v5/v9u combined5 ckpts (trained on
etoile+ZJU+SUTD+florence+munich) and evaluate on the new SHENZHEN test set
(diffuse scattering + 17k test samples, never seen during training).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet, tclnet, dcrnet_v20, dcrnet_v25, dcrnet_v25c, dcrnet_v29, dcrnet_v30a


def _v9u(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


MODEL_TAGS = {
    "v1": ("v1", lambda cr: dcrnet(reduction=cr, expansion=1), "combined5_v1_cr"),
    "v5": ("v5", lambda cr: dcrnet_v5(reduction=cr, expansion=1, ranks=16, r_enc=512), "combined5_v5_cr"),
    "v9u": ("v9", _v9u, "combined5_v9u_cr"),
    "transnet": ("transnet", lambda cr: transnet(reduction=cr, expansion=1), "combined5_transnet_cr"),
    "tclnet": ("tclnet", lambda cr: tclnet(reduction=cr, expansion=1), "combined5_tclnet_cr"),
    "v20": ("v20", lambda cr: dcrnet_v20(reduction=cr), "combined5_v20_cr"),
    "v25": ("v25", lambda cr: dcrnet_v25(reduction=cr), "combined5_v25_cr"),
    "v25c": ("v25c", lambda cr: dcrnet_v25c(reduction=cr), "combined5_v25c_cr"),
    "v29": ("v29", lambda cr: dcrnet_v29(reduction=cr), "combined5_v29_cr"),
    "v30a": ("v30a", lambda cr: dcrnet_v30a(reduction=cr), "combined5_v30a_cr"),
}


def _nmse(pred, gt):
    s = gt - 0.5
    p = pred - 0.5
    power = (s ** 2).sum(dim=[1, 2, 3])
    err = ((s - p) ** 2).sum(dim=[1, 2, 3])
    return 10 * torch.log10((err / power.clamp_min(1e-12)).mean()).item()


@torch.no_grad()
def _eval(model, loader, device):
    losses = []
    for batch in loader:
        x = batch[0].to(device)
        y = model(x)
        losses.append(_nmse(y, x))
    return float(np.mean(losses))


def main():
    device = "cuda:0"
    # Load new SHENZHEN test set
    sz = np.load("/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_SHENZHEN_test_3GHz_32x1024.npz",
                 allow_pickle=True)
    x_test = torch.from_numpy(sz["x_test"])
    print(f"SHENZHEN test: N={len(x_test)} (diffuse + scattering)")
    loader = DataLoader(TensorDataset(x_test), batch_size=200, shuffle=False)

    print(f"\n{'model':6s} {'cr':>4s}  shenzhen_test_NMSE  (combined5 in-domain mean)")
    print("-" * 65)

    # in-domain combined5 means (from earlier eval) for reference
    in_domain = {
        ("v1", 4): -12.57, ("v1", 8): -13.12, ("v1", 16): -11.93,
        ("v5", 4): -16.39, ("v5", 8): -13.54, ("v5", 16): -11.49,
        ("v9u", 4): -20.05, ("v9u", 8): -16.26, ("v9u", 16): -13.39,
        ("transnet", 4): -18.80, ("transnet", 8): -16.95, ("transnet", 16): -15.11,
        ("tclnet", 4): -20.02, ("tclnet", 8): -18.03, ("tclnet", 16): -15.01,
        ("v20", 4): -20.15, ("v20", 8): -16.77, ("v20", 16): -13.07,
        ("v25", 4): float("nan"), ("v25", 8): float("nan"), ("v25", 16): float("nan"),
        ("v25c", 4): -20.29, ("v25c", 8): -16.96, ("v25c", 16): -13.49,
        ("v29", 4): -20.68, ("v29", 8): -17.60, ("v29", 16): -13.37,
        ("v30a", 4): -20.13, ("v30a", 8): -16.16, ("v30a", 16): -12.18,
    }

    for mkey in ("v9u", "transnet", "tclnet", "v25c", "v29", "v30a"):
        for cr in (4, 8, 16):
            mpfx, factory, tag_pfx = MODEL_TAGS[mkey]
            ckpt = f"/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints/{mpfx}-1X-cr{cr}-in-{tag_pfx}{cr}"
            if not os.path.isfile(ckpt):
                print(f"{mkey:6s} {cr:>4d}  ckpt missing")
                continue
            sd = torch.load(ckpt, map_location=device)["state_dict"]
            model = factory(cr).to(device)
            model.load_state_dict(sd, strict=False)
            model.eval()
            n = _eval(model, loader, device)
            ind = in_domain.get((mkey, cr), float("nan"))
            gap = n - ind
            print(f"{mkey:6s} {cr:>4d}  {n:+8.3f}  ({ind:+.2f} in-domain, Δ={gap:+.2f})")


if __name__ == "__main__":
    main()
