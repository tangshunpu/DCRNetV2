"""COST2100 dataset utilities for DCRNetV2.

The public COST2100 CSI-feedback benchmark is expected to be stored as four
MATLAB files for each scenario (``in`` or ``out``)::

    DATA_Htrain{scenario}.mat        key: HT
    DATA_Hval{scenario}.mat          key: HT
    DATA_Htest{scenario}.mat         key: HT
    DATA_HtestF{scenario}_all.mat    key: HF_all

``HT`` is reshaped to ``(N, 2, 32, 32)`` and normalized in the same format used
by the original DCRNet/CsiNet benchmark. ``HF_all`` is complex full-bandwidth
CSI and is converted to ``(N, 32, 125, 2)`` for the rho metric.

This file can also be used as a small dataset-reading script::

    python read_dataset.py --data ./data/COST2100 --scenario in
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset


__all__ = [
    "Cost2100DataLoader",
    "PreFetcher",
    "load_cost2100_tensors",
    "summarize_cost2100",
]


SPARSE_SHAPE = (2, 32, 32)
RAW_SHAPE = (32, 125, 2)


class PreFetcher:
    """CUDA-stream async data prefetcher to overlap H2D copy with compute."""

    def __init__(self, loader: DataLoader):
        if not torch.cuda.is_available():
            raise RuntimeError("PreFetcher requires CUDA")
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self) -> None:
        try:
            batch = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # default_collate returns a list for TensorDataset tuples in modern
        # PyTorch, but converting keeps this robust across versions.
        batch = list(batch)
        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(batch):
                batch[idx] = tensor.cuda(non_blocking=True)
        self.next_input = batch

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_input
        if batch is None:
            raise StopIteration
        for tensor in batch:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return batch


def _load_mat_key(path: Path, keys: Iterable[str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    mat = sio.loadmat(path)
    for key in keys:
        if key in mat:
            return mat[key]
    available = ", ".join(k for k in mat.keys() if not k.startswith("__"))
    raise KeyError(f"None of keys {tuple(keys)} found in {path}. Available: {available}")


def _as_sparse_tensor(array: np.ndarray, *, name: str) -> torch.Tensor:
    array = np.asarray(array)
    if array.size % (2 * 32 * 32) != 0:
        raise ValueError(f"{name} cannot be reshaped to (N, 2, 32, 32); got {array.shape}")
    tensor = torch.as_tensor(array, dtype=torch.float32).view(-1, *SPARSE_SHAPE)
    return tensor.contiguous()


def _as_raw_tensor(array: np.ndarray, *, name: str) -> torch.Tensor:
    """Convert complex HF_all to (N, 32, 125, 2)."""

    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dims, got {array.shape}")

    n = array.shape[0]
    raw = np.asarray(array)
    # Handle 2D flattened (N, 4000) form: reshape to (N, 32, 125).
    if raw.ndim == 2:
        if raw.size % (n * 32 * 125) != 0:
            raise ValueError(f"{name} cannot be reshaped to (N, 32, 125); got {array.shape}")
        raw = raw.reshape(n, 32, 125)
    if raw.shape[1:3] == (32, 125):
        raw = raw[:, :32, :125]
    elif raw.shape[1:3] == (125, 32):
        raw = raw[:, :125, :32].transpose(0, 2, 1)
    else:
        # COST2100 releases may store HF_all as a flattened tail. Keep the
        # original DCRNet convention and reshape to antennas x subcarriers.
        if raw.size % (n * 32 * 125) != 0:
            raise ValueError(f"{name} cannot be reshaped to (N, 32, 125); got {array.shape}")
        raw = raw.reshape(n, 32, 125)

    real = torch.as_tensor(np.real(raw), dtype=torch.float32)
    imag = torch.as_tensor(np.imag(raw), dtype=torch.float32)
    return torch.stack((real, imag), dim=-1).contiguous()


def load_cost2100_tensors(root: str | os.PathLike, scenario: str) -> Tuple[torch.Tensor, ...]:
    """Load COST2100 tensors as ``train, val, test, raw_test``.

    Args:
        root: Directory containing the four COST2100 ``.mat`` files.
        scenario: ``"in"`` for indoor or ``"out"`` for outdoor.
    """

    root = Path(root).expanduser()
    if scenario not in {"in", "out"}:
        raise ValueError("scenario must be 'in' or 'out'")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    train = _as_sparse_tensor(
        _load_mat_key(root / f"DATA_Htrain{scenario}.mat", ("HT", "H_train", "train")),
        name="train HT",
    )
    val = _as_sparse_tensor(
        _load_mat_key(root / f"DATA_Hval{scenario}.mat", ("HT", "H_val", "val")),
        name="val HT",
    )
    test = _as_sparse_tensor(
        _load_mat_key(root / f"DATA_Htest{scenario}.mat", ("HT", "H_test", "test")),
        name="test HT",
    )
    raw_test = _as_raw_tensor(
        _load_mat_key(root / f"DATA_HtestF{scenario}_all.mat", ("HF_all", "HF", "H_test_full")),
        name="test HF_all",
    )
    if raw_test.shape[0] != test.shape[0]:
        raise ValueError(f"test/raw sample count mismatch: {test.shape[0]} vs {raw_test.shape[0]}")
    return train, val, test, raw_test


class Cost2100DataLoader:
    """Create train/val/test ``DataLoader`` objects for COST2100.

    The test loader yields ``(sparse_gt, raw_gt)`` pairs. ``raw_gt`` is used only
    for the rho metric; NMSE is computed on the sparse angular-delay tensor.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        batch_size: int = 200,
        num_workers: int = 4,
        pin_memory: bool = True,
        scenario: str = "in",
        prefetch: bool | None = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.prefetch = torch.cuda.is_available() if prefetch is None else bool(prefetch)

        train, val, test, raw_test = load_cost2100_tensors(root, scenario)
        self.train_dataset = TensorDataset(train)
        self.val_dataset = TensorDataset(val)
        self.test_dataset = TensorDataset(test, raw_test)

    def __call__(self):
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        train = DataLoader(self.train_dataset, shuffle=True, **kwargs)
        val = DataLoader(self.val_dataset, shuffle=False, **kwargs)
        test = DataLoader(self.test_dataset, shuffle=False, **kwargs)
        if self.pin_memory and self.prefetch and torch.cuda.is_available():
            train, val, test = PreFetcher(train), PreFetcher(val), PreFetcher(test)
        return train, val, test


def _tensor_stats(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tensor.min().item():.4g}, max={tensor.max().item():.4g}, "
        f"mean={tensor.float().mean().item():.4g}"
    )


def summarize_cost2100(root: str | os.PathLike, scenario: str) -> str:
    """Return a human-readable summary for a COST2100 split."""

    train, val, test, raw_test = load_cost2100_tensors(root, scenario)
    lines = [f"COST2100 summary: root={Path(root).expanduser()} scenario={scenario}"]
    lines.append(f"  train:    {_tensor_stats(train)}")
    lines.append(f"  val:      {_tensor_stats(val)}")
    lines.append(f"  test:     {_tensor_stats(test)}")
    lines.append(f"  raw_test: {_tensor_stats(raw_test)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and validate COST2100 .mat files for DCRNetV2")
    parser.add_argument("--data", default="./data/COST2100", help="COST2100 dataset root")
    parser.add_argument("--scenario", default="in", choices=["in", "out"], help="COST2100 scenario")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for loader smoke test")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers for smoke test")
    args = parser.parse_args()

    print(summarize_cost2100(args.data, args.scenario))
    train_loader, _, test_loader = Cost2100DataLoader(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        scenario=args.scenario,
        prefetch=False,
    )()
    train_batch = next(iter(train_loader))[0]
    test_sparse, test_raw = next(iter(test_loader))
    print(f"  loader train batch: {tuple(train_batch.shape)}")
    print(f"  loader test batch:  sparse={tuple(test_sparse.shape)}, raw={tuple(test_raw.shape)}")


if __name__ == "__main__":
    main()
