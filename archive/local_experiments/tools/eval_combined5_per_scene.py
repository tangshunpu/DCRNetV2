"""Evaluate a v1 combined5 checkpoint on each individual scene's test set.

For each (cr, scene), reports test NMSE in dB. Helps see whether joint training
on 5 scenes generalises uniformly or favours some scenes.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet, clnet, crnet, dcrnet_v20, dcrnet_v25, dcrnet_v25c, tclnet, dcrnet_v29, dcrnet_v30a

def _setenv(k, v):
    os.environ[k] = v
    return None


def _v12_rX(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=32, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=0.05,
    )


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
    "crnet": ("crnet", lambda cr: crnet(reduction=cr, expansion=1), "combined5_crnet_cr"),
    "clnet": ("clnet", lambda cr: clnet(reduction=cr, expansion=1), "combined5_clnet_cr"),
    "v9u_warm": ("v9", _v9u, "combined5_v9u_warm_cr"),
    "v9u_sase": ("v9", lambda cr: (_setenv("DCRNET_V9_SA", "1"),
                                    _setenv("DCRNET_V9_SE", "1"),
                                    _v9u(cr))[-1], "combined5_v9u_sase_cr"),
    "v12_rX": ("v9", _v12_rX, "combined5_v12rX_cr"),
    "v20": ("v20", lambda cr: dcrnet_v20(reduction=cr), "combined5_v20_cr"),
    "v25": ("v25", lambda cr: dcrnet_v25(reduction=cr), "combined5_v25_cr"),
    "v25c": ("v25c", lambda cr: dcrnet_v25c(reduction=cr), "combined5_v25c_cr"),
    "tclnet": ("tclnet", lambda cr: tclnet(reduction=cr, expansion=1), "combined5_tclnet_cr"),
    "v29": ("v29", lambda cr: dcrnet_v29(reduction=cr), "combined5_v29_cr"),
    "v30a": ("v30a", lambda cr: dcrnet_v30a(reduction=cr), "combined5_v30a_cr"),
}


SCENES = ["etoile", "ZJU", "SUTD", "florence", "munich"]
DATA_DIR = "/home/ubuntu/Documents/project/RT_CSI/outputs/csi"


def _nmse(pred, gt):
    s = gt - 0.5
    p = pred - 0.5
    power = (s ** 2).sum(dim=[1, 2, 3])
    err = ((s - p) ** 2).sum(dim=[1, 2, 3])
    return 10 * torch.log10((err / power.clamp_min(1e-12)).mean()).item()


@torch.no_grad()
def _eval(model, loader, device):
    losses = []
    for (x,) in loader:
        x = x.to(device)
        y = model(x)
        losses.append(_nmse(y, x))
    return float(np.mean(losses))


def eval_cr(model_key, cr, device="cuda:0"):
    model_pfx, factory, tag_pfx = MODEL_TAGS[model_key]
    ckpt_path = f"/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints/{model_pfx}-1X-cr{cr}-in-{tag_pfx}{cr}"
    if not os.path.isfile(ckpt_path):
        print(f"{model_key} cr={cr}: ckpt missing ({ckpt_path}), skip")
        return None
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    model = factory(cr).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    best_ep = torch.load(ckpt_path, map_location="cpu").get("epoch", "?")
    print(f"\n=== {model_key} combined5 cr={cr} (best ckpt @ep{best_ep}) ===")
    overall = []
    for scene in SCENES:
        npz = f"{DATA_DIR}/csi_{scene}_3GHz_32x1024.npz"
        z = np.load(npz, allow_pickle=True)
        x_test = torch.from_numpy(z["x_test"])
        loader = DataLoader(TensorDataset(x_test), batch_size=200, shuffle=False)
        n = _eval(model, loader, device)
        print(f"  {scene:9s} test NMSE = {n:+.3f} dB  (N={len(x_test)})")
        overall.append(n)
    print(f"  mean over 5 scenes: {np.mean(overall):+.3f}  spread={max(overall)-min(overall):.3f}")


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["v1", "v5"]
    for m in models:
        for cr in (4, 8, 16):
            eval_cr(m, cr)


if __name__ == "__main__":
    main()
