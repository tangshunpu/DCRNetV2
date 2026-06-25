"""RT_CSI (ZJU/SUTD + Sionna RT city scenes) CSI dataset loader.

Reads npz files produced by /home/ubuntu/Documents/project/RT_CSI scripts.
Same on-disk format as DeepMIMODataLoader (x_train/x_val/x_test in [0,1]).

Available scenes (under RT_CSI/outputs/csi):
    zju             csi_ZJU_3GHz_32x1024.npz
    munich          csi_munich_3GHz_32x1024.npz
    etoile          csi_etoile_3GHz_32x1024.npz
    florence        csi_florence_3GHz_32x1024.npz
    canyon          csi_simple_street_canyon_3GHz_32x1024.npz

All scenes: 3.5 GHz, 1024 sub, 32-ant ULA, angular-delay (2, 32, 32).
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .cost2100 import PreFetcher

__all__ = ['RTCSIDataLoader']


_SCENE_NPZ = {
    'zju':      'csi_ZJU_3GHz_32x1024.npz',
    'munich':   'csi_munich_3GHz_32x1024.npz',
    'etoile':   'csi_etoile_3GHz_32x1024.npz',
    'florence': 'csi_florence_3GHz_32x1024.npz',
    'canyon':   'csi_simple_street_canyon_3GHz_32x1024.npz',
    'sutd':     'csi_SUTD_3GHz_32x1024.npz',
    'shenzhen': 'csi_SHENZHEN_3GHz_32x1024.npz',
    'zjubs0':   'csi_zjuBS0_3GHz_32x1024.npz',
    'combined5': 'csi_combined5_3GHz_32x1024.npz',
}


class RTCSIDataLoader:
    """PyTorch DataLoader for RT_CSI Sionna-generated CSI npz files."""

    def __init__(self, root: str, batch_size: int, num_workers: int,
                 pin_memory: bool, scenario: str = 'in',  # unused
                 scene: str = 'zju'):
        del scenario
        assert scene in _SCENE_NPZ, f'unknown scene {scene}, expected {list(_SCENE_NPZ)}'
        path = os.path.join(root, _SCENE_NPZ[scene])
        assert os.path.isfile(path), f'missing {path}'

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.scene = scene

        z = np.load(path, allow_pickle=True)
        x_tr = torch.from_numpy(z['x_train'])
        x_va = torch.from_numpy(z['x_val'])
        x_te = torch.from_numpy(z['x_test'])
        # DCRNET_CENTERED=1: shift data from [0,1] convention (x = s/2 + 0.5)
        # to centered domain (s, mean=0). Used together with identity output
        # head + centered NMSE evaluator. See option B in RT_CSI_SCENES.md.
        if os.environ.get('DCRNET_CENTERED', '0') == '1':
            x_tr = (x_tr - 0.5) * 2.0
            x_va = (x_va - 0.5) * 2.0
            x_te = (x_te - 0.5) * 2.0
        self.scale = float(z['scale']) if 'scale' in z.files else float('nan')
        self.shape = tuple(x_tr.shape[1:])              # (2, 32, 32)
        self.n_train, self.n_val, self.n_test = len(x_tr), len(x_va), len(x_te)

        self.train_dataset = TensorDataset(x_tr)
        self.val_dataset = TensorDataset(x_va)
        # Mirror Rice / DeepMIMO test signature: (sparse_gt, raw_gt).
        # raw_gt unused for RT_CSI — pass sparse_gt twice.
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
