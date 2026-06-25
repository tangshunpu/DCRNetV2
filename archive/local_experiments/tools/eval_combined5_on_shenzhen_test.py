"""Cross-domain test: load all 8 combined5-trained ckpts and evaluate on
the new SHENZHEN_test set (30 BS all-BS, no diffuse, 40k samples).

Models: clnet, crnet, v1, v5, v9u, v9u_sase, v9u_warm, transnet
CRs:    4, 8, 16
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet, crnet, clnet, tclnet


def _v9u(cr):
    return dcrnet_v9(
        reduction=cr, expansion=1, r_enc=512,
        refine_width=2, refine_K=1,
        use_film=False, enc_fuse_mlp=64, complex_encoder=True,
        hybrid_decoder=True,
        hybrid_direct_ranks=32, hybrid_extra_ranks=16, hybrid_extra_d_emb=16,
        hybrid_gate_bias=-2.0, hybrid_alpha_init=1e-2,
    )


def _v9u_sase(cr):
    # SA + SE blocks gated via env vars (read by dcrnet_v9.__init__)
    os.environ["DCRNET_V9_SA"] = "1"
    os.environ["DCRNET_V9_SE"] = "1"
    m = _v9u(cr)
    return m


def _make_with_env(env_changes, factory, cr):
    """Build a model with temporary env-var overrides (DCRNet-v9 reads
    DCRNET_V9_* at __init__ time)."""
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


# (model_ckpt_prefix, ckpt_tag_template, factory) -- tag forms the suffix
# after `cr<N>` in the checkpoint filename.
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
    ("tclnet",    "tclnet",   lambda cr: tclnet(reduction=cr, expansion=1),   "combined5_tclnet_cr"),
]


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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sz = np.load("/home/ubuntu/Documents/project/RT_CSI/outputs/csi/csi_SHENZHEN_test_3GHz_32x1024.npz",
                 allow_pickle=True)
    x_test = torch.from_numpy(sz["x_test"])
    print(f"SHENZHEN_test: N={len(x_test)} samples (30 BS, all-BS, no diffuse)")
    loader = DataLoader(TensorDataset(x_test), batch_size=200, shuffle=False)

    print(f"\n{'model':10s} {'cr':>4s}  {'NMSE [dB]':>11s}  ckpt")
    print("-" * 80)
    rows = []
    ckpt_dir = "/home/ubuntu/Documents/project/DCRNet-V2/outputs/checkpoints"
    for mkey, mpfx, factory, tag_pfx in MODELS:
        for cr in (4, 8, 16):
            ckpt = f"{ckpt_dir}/{mpfx}-1X-cr{cr}-in-{tag_pfx}{cr}"
            if not os.path.isfile(ckpt):
                print(f"{mkey:10s} {cr:>4d}  {'--':>11s}  MISSING  ({ckpt})")
                rows.append((mkey, cr, None))
                continue
            try:
                sd = torch.load(ckpt, map_location=device,
                                weights_only=False)["state_dict"]
                model = factory(cr).to(device)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                model.eval()
                n = _eval(model, loader, device)
                print(f"{mkey:10s} {cr:>4d}  {n:+11.3f}  ok"
                      f"  (miss={len(missing)} unexp={len(unexpected)})")
                rows.append((mkey, cr, n))
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{mkey:10s} {cr:>4d}  ERROR: {type(e).__name__}: {e}")
                rows.append((mkey, cr, None))

    # Final clean table
    print("\n" + "=" * 60)
    print(f"{'model':10s}   {'cr=4':>9s}   {'cr=8':>9s}   {'cr=16':>9s}")
    print("-" * 60)
    table = {}
    for mkey, cr, n in rows:
        table[(mkey, cr)] = n
    for mkey, _, _, _ in MODELS:
        cells = []
        for cr in (4, 8, 16):
            v = table.get((mkey, cr))
            cells.append(f"{v:+.3f}" if v is not None else "    --")
        print(f"{mkey:10s}   {cells[0]:>9s}   {cells[1]:>9s}   {cells[2]:>9s}")


if __name__ == "__main__":
    main()
