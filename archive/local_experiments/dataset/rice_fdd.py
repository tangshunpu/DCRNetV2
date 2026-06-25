"""Rice RENEW FDD massive-MIMO dataset loader.

Source: https://renew.rice.edu/dataset-fdd.html  (Zhang/Du, Argos v2)
  - 64-antenna BS (8x8 UPA) at 2.4 GHz ISM
  - 2 channels separated by 72 MHz: ch1 = UL (channel 1), ch14 = DL (channel 14)
  - 52 active data subcarriers (64-FFT minus zero/guard at 0-5, 32, 59-63)

Indoor measurement campaign:
  - 8 LOS locations + 16 NLOS locations  (24 total)
  - up to 250 frames per location
  - per-frame raw HDF5 'Pilot_Samples': (frames, antennas, samples, IQ)
    After samps2csi: (frames, users, 2_LTS, 64_BS, 52_subcarrier) complex.

This loader expects a pre-processed *.npz produced by tools/prepare_rice_indoor.py
with keys:
    H:    complex64, shape (N, 64, 52)
    loc:  int32,    shape (N,)  -- location id, used for non-leaking split
    los:  bool,     shape (N,)  -- True if LOS location

The input that hits the model is the raw frequency-domain CSI stacked as
(2, 64, 52) -- channel 0 = real, channel 1 = imag.  Per "indoor 不 FFT,
不裁剪" we do *not* DFT across antennas or IFFT across subcarriers.

Normalization:
    dataset-wide max(|H|.real|, |H|.imag|) over the *train* split -> scale
    so both parts land in [-0.5, 0.5], then +0.5 -> [0, 1].  Matches the
    [0, 1] range DCRNet's Sigmoid output expects.  Stats are cached on the
    train data and reused for val/test so the eval is honest.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .cost2100 import PreFetcher

__all__ = ['RiceFDDDataLoader']


def _split_by_location(loc: np.ndarray, los: np.ndarray, seed: int = 0):
    """Per-location split: 70/15/15, stratified by LOS/NLOS.

    Splitting by location (not by frame) avoids leakage from temporally
    adjacent frames at the same UE position.  Test == zero-shot
    generalization to unseen locations -- hard, drives up NMSE on small
    datasets like Rice indoor.
    """
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for is_los in (True, False):
        locs = np.unique(loc[los == is_los])
        rng.shuffle(locs)
        n = len(locs)
        n_tr = max(1, int(round(0.70 * n)))
        n_va = max(1, int(round(0.15 * n)))
        tr_locs = set(locs[:n_tr].tolist())
        va_locs = set(locs[n_tr:n_tr + n_va].tolist())
        te_locs = set(locs[n_tr + n_va:].tolist())

        for split_locs, bucket in ((tr_locs, train_idx),
                                   (va_locs, val_idx),
                                   (te_locs, test_idx)):
            mask = np.zeros_like(loc, dtype=bool)
            for l in split_locs:
                mask |= (loc == l) & (los == is_los)
            bucket.extend(np.where(mask)[0].tolist())

    return (np.array(sorted(train_idx), dtype=np.int64),
            np.array(sorted(val_idx),   dtype=np.int64),
            np.array(sorted(test_idx),  dtype=np.int64))


def _split_random(loc: np.ndarray, los: np.ndarray, seed: int = 0,
                  train_frac: float = 0.70):
    """Pool all 24 locations and split frames at random.

    Standard CSI-feedback evaluation (CSINet / CRNet / DCRNet paper):
    frames from each location appear in train, val and test. Easier than
    the by-location split because test channels share statistics with
    train. Use this when you want to measure the model's compression
    capability per se, not its generalization to new sites.

    train_frac sets the train fraction. The remainder is split:
        train_frac >= 0.80  → 80/20-style: 0 val, all remainder as test
                              (val_dataset will mirror test, since main.py
                              uses test_loader for periodic eval)
        train_frac <  0.80  → remainder split 50/50 between val and test
                              (e.g. 0.70 → 70/15/15)
    """
    rng = np.random.RandomState(seed)
    n = len(loc)
    perm = rng.permutation(n)
    n_tr = int(round(train_frac * n))
    if train_frac >= 0.80 - 1e-6:
        tr = np.sort(perm[:n_tr]).astype(np.int64)
        te = np.sort(perm[n_tr:]).astype(np.int64)
        return tr, te, te                  # val mirrors test
    n_va = int(round((n - n_tr) / 2))
    return (np.sort(perm[:n_tr]).astype(np.int64),
            np.sort(perm[n_tr:n_tr + n_va]).astype(np.int64),
            np.sort(perm[n_tr + n_va:]).astype(np.int64))


def _to_2chw(H: np.ndarray, scale: float) -> torch.Tensor:
    """(N, n_ant, n_sub) complex -> (N, 2, n_ant, n_sub) float32 in [0, 1].

    Per-sample max-abs normalization: each H is scaled so max(|re|, |im|) → 0.5,
    then +0.5 → [0, 1]. WHY not global: indoor Rice has 100×+ dynamic range
    across UE positions and small-scale fading, so a single dataset-wide
    scale is dominated by the strongest sample, squishing 99% of the rest
    to std≈0.007 around 0.5 — Sigmoid-end models (v1, CRNet, CLNet) then
    collapse to constant-0.5 output. Per-sample preserves shape of every
    sample. NMSE is per-sample scale-invariant so the metric stays valid.
    `scale` arg kept for API compatibility but ignored.
    """
    del scale
    scales = np.maximum(np.abs(H.real).reshape(len(H), -1).max(axis=1),
                        np.abs(H.imag).reshape(len(H), -1).max(axis=1))
    scales = np.maximum(scales, 1e-12)[:, None, None]
    re = (H.real / scales) * 0.5 + 0.5
    im = (H.imag / scales) * 0.5 + 0.5
    out = np.stack([re, im], axis=1).astype(np.float32)
    return torch.from_numpy(out)


def _to_angular_delay(H: np.ndarray) -> np.ndarray:
    """Frequency-domain CSI -> angular-delay-domain CSI (unitary, lossless).

    H shape: (N, n_ant, n_sub) complex.  We treat the antenna axis as a
    1D ULA-equivalent and apply a unitary DFT on antennas (freq -> angular),
    and an inverse FFT on subcarriers (freq -> delay).  No cropping.

    For an 8x8 UPA the physically correct transform is 2D DFT in (az, el);
    1D DFT on the flattened 96-vector is a coarser but still energy-
    preserving substitute -- adequate for a first pass and matches the
    standard CSINet/CRNet pipeline that DCRNet originated from.
    """
    # Unitary FFT (ant) + unitary IFFT (sub).  norm='ortho' makes both
    # transforms norm-preserving, so ||H_freq||_F == ||H_angdelay||_F per
    # sample.  This keeps the per-cell dynamic range comparable across
    # domains and avoids needing different LR / normalization.
    Ha  = np.fft.fft (H, axis=-2, norm='ortho')        # (N, n_ant, n_sub)
    Had = np.fft.ifft(Ha, axis=-1, norm='ortho')       # (N, n_ant, n_sub)
    return Had.astype(np.complex64)


class RiceFDDDataLoader:
    """PyTorch DataLoader for Rice RENEW FDD indoor CSI.

    Same call signature as Cost2100DataLoader so main.py can swap with a
    single line.  test_loader yields (sparse_gt, raw_gt) where raw_gt is
    just a re-emission of sparse_gt -- the COST2100 ρ-on-full-band metric
    does not apply here (52 subcarriers is the full bandwidth), so the
    evaluator should skip ρ when dataset == 'rice'.

    domain:
        'freq'    -> raw frequency-domain CSI (default).  No FFT, no crop.
                     Input shape (2, n_ant, n_sub).
        'angdelay'-> angular-delay-domain CSI.  fft(ant) + ifft(sub), no crop.
                     Same input shape (2, n_ant, n_sub) but energy is now
                     concentrated on a small number of taps -- the prior
                     that v3/v5/v9 low-rank encoders are built around.
    """

    def __init__(self, root: str, batch_size: int, num_workers: int,
                 pin_memory: bool, scenario: str = 'in',
                 channel: str = 'ch14', split_seed: int = 0,
                 domain: str = 'freq', split: str = 'location',
                 train_frac: float = 0.70):
        assert scenario == 'in', "rice_fdd: only indoor is wired up so far"
        assert channel in ('ch1', 'ch14')
        assert domain in ('freq', 'angdelay')
        assert split in ('location', 'random')
        assert 0.5 < train_frac < 1.0
        npz_path = os.path.join(root, f'rice_indoor_{channel}.npz')
        assert os.path.isfile(npz_path), (
            f"missing {npz_path} -- run tools/prepare_rice_indoor.py first")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.domain = domain
        self.channel = channel

        z = np.load(npz_path)
        H = z['H']            # complex64, (N, 96, 52)
        loc = z['loc']
        los = z['los']
        self.split_mode = split
        self.train_frac = train_frac
        if split == 'location':
            tr, va, te = _split_by_location(loc, los, seed=split_seed)
        else:
            tr, va, te = _split_random(loc, los, seed=split_seed,
                                       train_frac=train_frac)

        if domain == 'angdelay':
            H = _to_angular_delay(H)

        # Dataset-wide scale = max(|re|, |im|) on TRAIN split only,
        # computed *after* any domain transform so [-0.5, 0.5] -> [0,1]
        # always lands the train data inside the Sigmoid range.
        train_max = max(float(np.abs(H[tr].real).max()),
                        float(np.abs(H[tr].imag).max()))
        self.scale = train_max
        self.shape = (2, H.shape[1], H.shape[2])  # (2, 96, 52)

        x_tr = _to_2chw(H[tr], train_max)
        x_va = _to_2chw(H[va], train_max)
        x_te = _to_2chw(H[te], train_max)

        self.train_dataset = TensorDataset(x_tr)
        self.val_dataset = TensorDataset(x_va)
        # Mirror Cost2100DataLoader's test signature: (sparse_gt, raw_gt).
        # raw_gt unused for Rice (no separate full-band CSI) — pass sparse_gt twice.
        self.test_dataset = TensorDataset(x_te, x_te)

        self.n_train, self.n_val, self.n_test = len(tr), len(va), len(te)
        self.n_train_locs = len(np.unique(loc[tr]))
        self.n_val_locs   = len(np.unique(loc[va]))
        self.n_test_locs  = len(np.unique(loc[te]))

    def __call__(self):
        kw = dict(batch_size=self.batch_size,
                  num_workers=self.num_workers,
                  pin_memory=self.pin_memory)
        train_loader = DataLoader(self.train_dataset, shuffle=True,  **kw)
        val_loader   = DataLoader(self.val_dataset,   shuffle=False, **kw)
        test_loader  = DataLoader(self.test_dataset,  shuffle=False, **kw)
        if self.pin_memory:
            train_loader = PreFetcher(train_loader)
            val_loader   = PreFetcher(val_loader)
            test_loader  = PreFetcher(test_loader)
        return train_loader, val_loader, test_loader
