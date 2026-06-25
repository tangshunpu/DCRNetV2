"""Cross-domain test: combined5 ckpts on Rice outdoor (subset).

Loads a few Rice outdoor HDF5 files in memory, processes them like
prepare_rice_indoor.py (samps2csi → 64-FFT → 52 active sub), applies per-
sample max-abs norm + angular-delay transform + 32x32 crop, then evaluates
v1/v5/v9u/transnet combined5 checkpoints.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import dcrnet, dcrnet_v5, dcrnet_v9, transnet, dcrnet_v20, dcrnet_v25

# borrow samps2csi + constants from the indoor prep script
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from prepare_rice_indoor import samps2csi


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
    "v20": ("v20", lambda cr: dcrnet_v20(reduction=cr),                          "combined5_v20_cr"),
    "v25": ("v25", lambda cr: dcrnet_v25(reduction=cr),                          "combined5_v25_cr"),
}


def load_outdoor_subset(channel: str = "ch14", n_los: int = 3, n_nlos: int = 3) -> np.ndarray:
    """Load a few LOS + NLOS Rice outdoor HDF5 files and process to complex CSI."""
    src = "/home/ubuntu/Documents/dataset/RICE_RENEW_FDD/outdoor-fdd/channel14"
    los_files = sorted(glob.glob(os.path.join(src, f"{channel}_los_loc*.hdf5")))[:n_los]
    nlos_files = sorted(glob.glob(os.path.join(src, f"{channel}_nlos_loc*.hdf5")))[:n_nlos]
    files = los_files + nlos_files
    print(f"loading {len(files)} outdoor HDF5 files (LOS={n_los}, NLOS={n_nlos})")
    Hs = []
    for path in files:
        t0 = time.time()
        with h5py.File(path, "r") as f:
            samps = f["Pilot_Samples"][:]
            samps_per_user = int(f.attrs["samples_per_user"])
            num_mob_ant = int(f.attrs["num_mob_ant"])
        csi = samps2csi(samps, num_users_plus1=num_mob_ant + 1,
                        samps_per_user=samps_per_user)
        # csi: (frames, n_users+1, 2_LTS, n_bs, 52) → keep ue=0, LTS=0
        H = csi[:, 0, 0, :, :].astype(np.complex64)  # (frames, n_bs, 52)
        Hs.append(H)
        print(f"  {os.path.basename(path)}: {H.shape} {time.time()-t0:.1f}s")
    return np.concatenate(Hs, axis=0)


def freq_to_angdelay(H: np.ndarray) -> np.ndarray:
    """(N, n_ant, n_sub) complex → (N, n_ant, n_sub) complex angular-delay.
    fft over antennas, ifft over subcarriers; both unitary (norm='ortho')."""
    H_ang = np.fft.fft(H, axis=1, norm="ortho")
    H_ad = np.fft.ifft(H_ang, axis=2, norm="ortho")
    return H_ad.astype(np.complex64)


def per_sample_norm(H: np.ndarray) -> np.ndarray:
    """Complex → (N, 2, n_ant, n_sub) float32 in [0, 1] per-sample max-abs."""
    flat = H.reshape(len(H), -1)
    scales = np.maximum(np.abs(flat.real).max(axis=1),
                        np.abs(flat.imag).max(axis=1))
    scales = np.maximum(scales, 1e-12)[:, None, None]
    re = (H.real / scales) * 0.5 + 0.5
    im = (H.imag / scales) * 0.5 + 0.5
    return np.stack([re, im], axis=1).astype(np.float32)


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
    H = load_outdoor_subset("ch14", n_los=3, n_nlos=3)
    print(f"\ntotal samples: {H.shape[0]}  ant={H.shape[1]}  sub={H.shape[2]}")
    print("applying angular-delay transform + per-sample norm...")
    H_ad = freq_to_angdelay(H)
    x = per_sample_norm(H_ad)
    print(f"x shape after norm: {x.shape}  range=[{x.min():.4f}, {x.max():.4f}]")

    # Crop top-left to (2, 32, 32) — energy-concentrated for ang-delay
    x_crop = x[:, :, :32, :32].copy()
    print(f"cropped to (2, 32, 32):  std={x_crop.std():.4f}")

    device = "cuda:0"
    x_te = torch.from_numpy(x_crop)
    loader = DataLoader(TensorDataset(x_te), batch_size=200, shuffle=False)

    in_domain = {
        ("v1", 4): -12.57, ("v1", 8): -13.12, ("v1", 16): -11.93,
        ("v5", 4): -16.39, ("v5", 8): -13.54, ("v5", 16): -11.49,
        ("v9u", 4): -20.05, ("v9u", 8): -16.26, ("v9u", 16): -13.39,
        ("transnet", 4): -18.80, ("transnet", 8): -16.95, ("transnet", 16): -15.11,
    }

    print(f"\n{'model':10s} {'cr':>4s}  rice_outdoor_NMSE  (combined5 in-domain)")
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
            n = _eval(model, loader, device)
            ind = in_domain.get((mkey, cr), float("nan"))
            print(f"{mkey:10s} {cr:>4d}  {n:+8.3f}  ({ind:+.2f}, Δ={n-ind:+.2f})")


if __name__ == "__main__":
    main()
