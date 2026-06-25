"""Convert Rice RENEW FDD indoor HDF5 raw traces to a flat .npz the
DCRNet RiceFDDDataLoader can consume.

Usage:
    python tools/prepare_rice_indoor.py \
        --src  ~/Documents/dataset/RICE_RENEW_FDD/indoor/ \
        --dst  ~/Documents/dataset/RICE_RENEW_FDD/ \
        --channel ch14        # ch1 = UL, ch14 = DL

Per-location HDF5 files are named  ch{1,14}_{los,nlos}_loc{N}.hdf5  with
Pilot_Samples of shape (frames, antennas, samples, IQ-2).  After
samps2csi (per the Argos v2 reference script) we get a complex tensor of
shape (frames, num_users+1, 2_LTS, 64_BS, 52_subcarrier).  We keep:
  - user index 0  (the UE we actually care about)
  - LTS index 0   (use the first of two estimates — matches the script's
                   userCSI = csi[:,:,0,:,:])

Final per-sample shape: (64, 52) complex64.  We concatenate over all
indoor locations and frames, and record the per-sample location id + a
boolean LOS flag.  Output: rice_indoor_<channel>.npz with keys H, loc, los.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np


# lts_freq comes from python_argosv2/argoscc/lts.py — inlined here to avoid
# importing that whole package (which pulls in interp1d etc).  This is the
# 64-point frequency-domain Long Training Sequence pattern.
LTS_FREQ = np.array(
    [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1,
     1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
     0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1,
     1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    dtype=np.float64,
)
ZERO_SUBC = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]   # 64 -> 52 active


def samps2csi(samps: np.ndarray, num_users_plus1: int,
              samps_per_user: int, offset: int = 47, fft_size: int = 64,
              chunk: int = 1000):
    """Port of argos_trace_check_xing.samps2csi.

    Args:
        samps:           (frames, BS_ant, samples, IQ) int16-ish array
                         where samples = num_users_plus1 * samps_per_user
        num_users_plus1: f.attrs['num_mob_ant'] + 1   (+1 for noise slot)
        samps_per_user:  f.attrs['samples_per_user']  (=224 for indoor)
        offset:          per-user offset into the samps_per_user window
                         where the first LTS starts. The reference script
                         uses 15+32 = 47 (cp + first LTS = 64 then second
                         LTS at +32 more samples). offset+128 must fit
                         within samps_per_user.
    Returns:
        csi: complex64 (frames, num_users_plus1, 2_LTS, BS_ant, 52_subcarrier)
    """
    n_fr, n_bs, _, _ = samps.shape
    out = np.empty((n_fr, num_users_plus1, 2, n_bs, 52), dtype=np.complex64)
    for s in range(0, n_fr, chunk):
        e = min(s + chunk, n_fr)
        block = samps[s:e]   # (b, n_bs, n_users+1 * samps_per_user, 2)
        # Reference reshape: (frames, n_bs, n_users+1, samps_per_user, 2)
        us = np.reshape(
            block, (block.shape[0], n_bs, num_users_plus1, samps_per_user, 2))
        iq = np.empty((block.shape[0], n_bs, num_users_plus1, 2, fft_size),
                      dtype=np.complex64)
        for i in range(2):       # two LTS estimates per user per frame
            sl = slice(offset + i * fft_size, offset + (i + 1) * fft_size)
            iq[:, :, :, i, :] = (us[:, :, :, sl, 0]
                                  + 1j * us[:, :, :, sl, 1]) * (2 ** -15)
        # (frames, n_bs, n_users+1, 2_LTS, fft) -> (frames, n_users+1, 2_LTS, n_bs, fft)
        iq = iq.swapaxes(1, 2).swapaxes(2, 3)
        csi = np.fft.fftshift(np.fft.fft(iq, fft_size, axis=4), axes=4) * LTS_FREQ
        csi = np.delete(csi, ZERO_SUBC, axis=4)
        out[s:e] = csi
    return out


_loc_re = re.compile(r'^(ch\d+)_(los|nlos)_loc(\d+)\.hdf5$', re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='dir containing ch*_loc*.hdf5 files (indoor extract)')
    ap.add_argument('--dst', required=True, help='output dir')
    ap.add_argument('--channel', choices=['ch1', 'ch14'], default='ch14',
                    help='ch1 = UL (channel 1), ch14 = DL (channel 14)')
    ap.add_argument('--ue', type=int, default=0,
                    help='UE index to keep (default 0)')
    args = ap.parse_args()

    pattern = os.path.join(args.src, f'{args.channel}_*_loc*.hdf5')
    files = sorted(glob.glob(pattern))
    if not files:
        # also try recursive — extract may put hdf5 in a subdir
        files = sorted(glob.glob(
            os.path.join(args.src, '**', f'{args.channel}_*_loc*.hdf5'),
            recursive=True))
    if not files:
        sys.exit(f'no HDF5 files matching {pattern}')

    print(f'found {len(files)} HDF5 file(s) for {args.channel}')

    Hs, locs, loss = [], [], []
    for path in files:
        base = os.path.basename(path)
        m = _loc_re.match(base)
        if not m:
            print(f'  skip (unrecognized name): {base}'); continue
        _, los_str, loc_str = m.groups()
        loc_id = int(loc_str)
        is_los = (los_str.lower() == 'los')

        t0 = time.time()
        with h5py.File(path, 'r') as f:
            samps = f['Pilot_Samples'][:]                     # full load
            samps_per_user = int(f.attrs['samples_per_user'])
            num_mob_ant = int(f.attrs['num_mob_ant'])
        # samps[..., axis 1] holds (num_mob_ant + 1_noise) * samps_per_user
        # + maybe pilot/calibration; samps2csi splits by num_users+1.
        csi = samps2csi(samps, num_users_plus1=num_mob_ant + 1,
                        samps_per_user=samps_per_user)
        # csi: (frames, n_users+1, 2_LTS, 64_BS, 52_sub)  -> keep ue, LTS=0
        H = csi[:, args.ue, 0, :, :]                          # (frames, 64, 52)
        Hs.append(H.astype(np.complex64))
        locs.append(np.full(H.shape[0], loc_id, dtype=np.int32))
        loss.append(np.full(H.shape[0], is_los, dtype=bool))
        print(f'  {base:32s}  frames={H.shape[0]:4d}  '
              f'|H| in [{np.abs(H).min():.3g}, {np.abs(H).max():.3g}]  '
              f'({time.time()-t0:.1f}s)')

    H = np.concatenate(Hs, axis=0)
    loc = np.concatenate(locs, axis=0)
    los = np.concatenate(loss, axis=0)
    print(f'\ntotal: H={H.shape} loc={loc.shape} los={los.shape}')
    print(f'  LOS samples: {los.sum()} ({len(np.unique(loc[los]))} locations)')
    print(f'  NLOS samples: {(~los).sum()} ({len(np.unique(loc[~los]))} locations)')

    os.makedirs(args.dst, exist_ok=True)
    out_path = os.path.join(args.dst, f'rice_indoor_{args.channel}.npz')
    np.savez_compressed(out_path, H=H, loc=loc, los=los)
    print(f'\nwrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
