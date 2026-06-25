"""Cross-domain test: COST2100-trained v5/v9u ckpts evaluated on combined5
test set (etoile/ZJU/SUTD/florence/munich).

COST2100 = canonical synthetic indoor CSI benchmark (lossy DFT into AD).
combined5 = Sionna-RT-generated outdoor scenes (different physics).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet_v5, dcrnet_v9


def _v9u(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


COST2100_CKPTS = {
    # v5 default (no tag) — COST2100 indoor
    ("v5_cost", 4):  ("v5", lambda cr: dcrnet_v5(reduction=cr, expansion=1, ranks=16, r_enc=512), "v5-1X-cr4-in"),
    ("v5_cost", 8):  ("v5", lambda cr: dcrnet_v5(reduction=cr, expansion=1, ranks=16, r_enc=512), "v5-1X-cr8-in"),
    ("v5_cost", 16): ("v5", lambda cr: dcrnet_v5(reduction=cr, expansion=1, ranks=16, r_enc=512), "v5-1X-cr16-in"),
    # v9 unified (tag=v12r) — COST2100 indoor
    ("v9u_cost_in", 4):  ("v9", _v9u, "v9-1X-cr4-in-v12r"),
    ("v9u_cost_in", 8):  ("v9", _v9u, "v9-1X-cr8-in-v12r"),
    ("v9u_cost_in", 16): ("v9", _v9u, "v9-1X-cr16-in-v12r"),
    # v9 unified (tag=v12r) — COST2100 outdoor (more relevant to RT_CSI outdoor scenes)
    ("v9u_cost_out", 4):  ("v9", _v9u, "v9-1X-cr4-out-v12r"),
    ("v9u_cost_out", 8):  ("v9", _v9u, "v9-1X-cr8-out-v12r"),
    ("v9u_cost_out", 16): ("v9", _v9u, "v9-1X-cr16-out-v12r"),
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


SCENES = ["etoile", "ZJU", "SUTD", "florence", "munich"]
DATA_DIR = "/home/ubuntu/Documents/project/RT_CSI/outputs/csi"


def main():
    device = "cuda:0"
    print(f"\n{'model':10s} {'cr':>4s}  " + "  ".join(f"{s:>9s}" for s in SCENES) + f"  {'mean':>8s}")
    print("-" * 90)
    for (mkey, cr), (mpfx, factory, tag) in COST2100_CKPTS.items():
        ckpt = f"/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints/{tag}"
        if not os.path.isfile(ckpt):
            print(f"{mkey:10s} {cr:>4d}  ckpt missing ({tag})")
            continue
        sd = torch.load(ckpt, map_location=device)["state_dict"]
        model = factory(cr).to(device)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"{mkey:10s} {cr:>4d}  load failed: {e}")
            continue
        model.eval()
        per_scene = []
        for scene in SCENES:
            npz = f"{DATA_DIR}/csi_{scene}_3GHz_32x1024.npz"
            z = np.load(npz, allow_pickle=True)
            x_te = torch.from_numpy(z["x_test"])
            loader = DataLoader(TensorDataset(x_te), batch_size=200, shuffle=False)
            n = _eval(model, loader, device)
            per_scene.append(n)
        mean = np.mean(per_scene)
        print(f"{mkey:10s} {cr:>4d}  " + "  ".join(f"{n:>+9.3f}" for n in per_scene) + f"  {mean:>+8.3f}")


if __name__ == "__main__":
    main()
