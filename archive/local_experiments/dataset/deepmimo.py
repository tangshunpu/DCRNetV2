"""DeepMIMO O1 / Outdoor1 (3.4 GHz) CSI feedback dataset loader.

Same call signature as Cost2100DataLoader / RiceFDDDataLoader so main.py
can swap with a single line.

Data file: deepmimo_o1_3p4.npz produced by tools/prepare_deepmimo_o1.py.
    x_train: (20000, 2, 32, 32)  float32  already in [0, 1]
    x_val:   (10000, 2, 32, 32)
    x_test:  (10000, 2, 32, 32)
    scale:   train-set max-abs (for de-normalization, if needed)

The angular-delay transform + max-abs normalization to [0, 1] happens at
prepare time; the loader is a thin wrapper that returns DataLoaders.

test_loader yields (sparse_gt, raw_gt) where raw_gt mirrors sparse_gt --
matches the Rice loader convention.  The evaluator should skip the
COST2100 ρ-on-full-band step when dataset == 'deepmimo'.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .cost2100 import PreFetcher

__all__ = ['DeepMIMODataLoader']


class DeepMIMODataLoader:
    """PyTorch DataLoader for DeepMIMO O1_3p4 (Outdoor1 NLoS) angular-delay CSI."""

    def __init__(self, root: str, batch_size: int, num_workers: int,
                 pin_memory: bool, scenario: str = 'in',  # unused, present for CLI symmetry
                 npz_name: str = 'deepmimo_o1_3p4.npz'):
        del scenario
        path = os.path.join(root, npz_name)
        assert os.path.isfile(path), (
            f"missing {path} -- run tools/prepare_deepmimo_o1.py first")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        z = np.load(path, allow_pickle=True)
        x_tr = torch.from_numpy(z['x_train'])
        x_va = torch.from_numpy(z['x_val'])
        x_te = torch.from_numpy(z['x_test'])
        self.scale = float(z['scale'])
        self.shape = tuple(x_tr.shape[1:])    # (2, 32, 32)
        self.n_train, self.n_val, self.n_test = len(x_tr), len(x_va), len(x_te)

        self.train_dataset = TensorDataset(x_tr)
        self.val_dataset = TensorDataset(x_va)
        # Mirror Rice / Cost2100 test signature: (sparse_gt, raw_gt).
        # raw_gt unused for DeepMIMO — pass sparse_gt twice.
        self.test_dataset = TensorDataset(x_te, x_te)

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
