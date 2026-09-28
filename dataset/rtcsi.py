"""Read RT-CSI release NPZ splits without depending on its generator package.

Pass the NPZ file directly. A separate test NPZ can be supplied for datasets
whose training archive has no test rows (for example, few-shot adaptation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .cost2100 import PreFetcher


CSI_SHAPE = (2, 32, 32)


def load_rtcsi_split(path: str | Path, split: str) -> torch.Tensor:
    """Load one normalized CSI split shaped ``(N, 2, 32, 32)``."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown RT-CSI split: {split}")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"RT-CSI NPZ not found: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Expected an RT-CSI .npz file, got: {path}")
    key = f"x_{split}"
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"Missing {key!r} in {path}; available keys: {archive.files}")
        array = archive[key]
    if array.ndim != 4 or tuple(array.shape[1:]) != CSI_SHAPE:
        raise ValueError(f"{path}:{key} must have shape (N, 2, 32, 32), got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path}:{key} must contain floating-point CSI, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}:{key} contains NaN or infinity")
    if array.size and (array.min() < -1e-5 or array.max() > 1.00001):
        raise ValueError(f"{path}:{key} must be normalized to [0, 1]")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def load_rtcsi_tensors(data_path: str | Path, test_path: str | Path | None = None):
    """Return train, validation, and test tensors from one or two NPZ files."""

    train = load_rtcsi_split(data_path, "train")
    val = load_rtcsi_split(data_path, "val")
    test = load_rtcsi_split(test_path or data_path, "test")
    if not len(train) or not len(val) or not len(test):
        raise ValueError("RT-CSI train/val/test must all be nonempty; use --test-data for a separate test NPZ")
    return train, val, test


class RTCSIDataLoader:
    """Create train/validation/test loaders for normalized RT-CSI CSI."""

    def __init__(
        self,
        data_path: str | Path,
        test_path: str | Path | None = None,
        batch_size: int = 200,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch: bool | None = None,
    ):
        train, val, test = load_rtcsi_tensors(data_path, test_path)
        self.train_dataset = TensorDataset(train)
        self.val_dataset = TensorDataset(val)
        self.test_dataset = TensorDataset(test)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.prefetch = torch.cuda.is_available() if prefetch is None else bool(prefetch)

    def __call__(self):
        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
        train = DataLoader(self.train_dataset, shuffle=True, **kwargs)
        val = DataLoader(self.val_dataset, shuffle=False, **kwargs)
        test = DataLoader(self.test_dataset, shuffle=False, **kwargs)
        if self.pin_memory and self.prefetch and torch.cuda.is_available():
            train, val, test = PreFetcher(train), PreFetcher(val), PreFetcher(test)
        return train, val, test
