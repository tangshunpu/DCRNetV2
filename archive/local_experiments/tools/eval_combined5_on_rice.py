"""Cross-domain test: v9u combined5 ckpts → Rice indoor real measured data.

Rice ch14, angdelay domain (2,32,32). True OOD: synthetic Sionna RT trained model
applied to real RENEW FDD measurements.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet
from dataset import RiceFDDDataLoader


def _v9u(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


MODELS = {
    "v1":  ("v1", lambda cr: dcrnet(reduction=cr, expansion=1),                "combined5_v1_cr"),
    "v5":  ("v5", lambda cr: dcrnet_v5(reduction=cr, expansion=1, ranks=16, r_enc=512), "combined5_v5_cr"),
    "v9u": ("v9", _v9u,                                                         "combined5_v9u_cr"),
    "transnet": ("transnet", lambda cr: transnet(reduction=cr, expansion=1),   "combined5_transnet_cr"),
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
        x = batch[0].to(device) if not isinstance(batch, list) else batch[0].to(device)
        y = model(x)
        losses.append(_nmse(y, x))
    return float(np.mean(losses))


def main():
    device = "cuda:0"
    # Rice indoor ch14 angdelay test loader
    rice = RiceFDDDataLoader(
        root="/home/ubuntu/Documents/dataset/RICE_RENEW_FDD",
        batch_size=200, num_workers=0, pin_memory=False,
        channel="ch14", domain="angdelay", split="random",
    )
    _, _, test_loader = rice()
    # test_loader yields (sparse_gt, raw_gt) tuples — only need sparse_gt for NMSE
    print(f"Rice indoor ch14 angdelay test: ~{rice.n_test} samples")

    in_domain = {
        ("v1", 4): -12.57, ("v1", 8): -13.12, ("v1", 16): -11.93,
        ("v5", 4): -16.39, ("v5", 8): -13.54, ("v5", 16): -11.49,
        ("v9u", 4): -20.05, ("v9u", 8): -16.26, ("v9u", 16): -13.39,
        ("transnet", 4): -18.80, ("transnet", 8): -16.95, ("transnet", 16): -15.11,
    }

    print(f"\n{'model':10s} {'cr':>4s}  rice_test_NMSE  (combined5 in-domain mean)")
    print("-" * 65)
    for mkey in ("v1", "v5", "v9u", "transnet"):
        for cr in (4, 8, 16):
            mpfx, factory, tag_pfx = MODELS[mkey]
            ckpt = f"/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints/{mpfx}-1X-cr{cr}-in-{tag_pfx}{cr}"
            if not os.path.isfile(ckpt):
                print(f"{mkey:10s} {cr:>4d}  ckpt missing")
                continue
            sd = torch.load(ckpt, map_location=device)["state_dict"]
            model = factory(cr).to(device)
            model.load_state_dict(sd, strict=False)
            model.eval()
            # Use a fresh loader since some test_loader has 2-tuple structure
            x_te = []
            for batch in test_loader:
                x_te.append(batch[0])
            x_te = torch.cat(x_te, dim=0)
            # Rice angdelay is (N, 2, 96, 52). combined5 models expect (2, 32, 32).
            # Resample with bilinear interp to preserve all info, vs cropping.
            if x_te.shape[-2:] != (32, 32):
                import torch.nn.functional as F
                # Convert [0,1] -> centered [-1,1], interp, then back
                xc = (x_te - 0.5) * 2
                xc = F.interpolate(xc, size=(32, 32), mode="bilinear", align_corners=False)
                x_te = (xc * 0.5 + 0.5).contiguous()
            simple_loader = DataLoader(TensorDataset(x_te), batch_size=200, shuffle=False)
            n = _eval(model, simple_loader, device)
            ind = in_domain.get((mkey, cr), float("nan"))
            print(f"{mkey:10s} {cr:>4d}  {n:+8.3f}  ({ind:+.2f} in-domain, Δ={n-ind:+.2f})")


if __name__ == "__main__":
    main()
