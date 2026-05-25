# DCRNetV2

DCRNetV2 is a lightweight CSI feedback compression model family for FDD massive MIMO-OFDM systems. This repository publishes a self-contained PyTorch release with model definitions, COST2100 dataset reading utilities, and training scripts.

## Highlights

- **Complex Hermitian low-rank encoder**: learns matched-filter measurements while preserving real/imaginary cross terms.
- **Hybrid low-rank decoder**: combines a direct low-rank decoder with a gated factored residual bank for better high-compression reconstruction.
- **Six ready-to-run variants**: `mini`, `small`, `base`, `unified`, `large-out`, and `large-in4` cover edge to high-accuracy settings.
- **COST2100 ready**: includes `.mat` readers, NMSE/rho evaluation, checkpointing, JSONL logs, and shell launchers.

## Repository layout

```text
.
├── dcrnetv2_release/      # Public release package
│   ├── dcrnetv2.py        # Self-contained DCRNetV2 model definitions
│   ├── dataset.py         # COST2100 loading + validation CLI
│   ├── train.py           # Training loop
│   ├── utils.py           # Scheduler, meters, NMSE/rho metrics
│   └── __init__.py
├── scripts/
│   ├── read_cost2100.sh   # Dataset smoke test launcher
│   └── train_cost2100.sh  # Training launcher
├── train.py               # Root wrapper: python train.py ...
├── read_dataset.py        # Root wrapper: python read_dataset.py ...
├── requirements.txt
└── AGENTS.md
```

> Note: datasets, checkpoints, logs, and experiment outputs are intentionally not tracked.

## Installation

```bash
git clone git@github.com:tangshunpu/DCRNetV2.git
cd DCRNetV2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation can vary by CUDA version. If needed, install the CUDA-specific wheel from the official PyTorch instructions first, then run `pip install -r requirements.txt`.

## Dataset

Download the COST2100 CSI-feedback benchmark used by CsiNet/DCRNet and place the `.mat` files under `./data/COST2100` or pass a custom path with `--data`.

Expected files for each scenario (`in` indoor, `out` outdoor):

```text
data/COST2100/
├── DATA_Htrainin.mat
├── DATA_Hvalin.mat
├── DATA_Htestin.mat
├── DATA_HtestFin_all.mat
├── DATA_Htrainout.mat
├── DATA_Hvalout.mat
├── DATA_Htestout.mat
└── DATA_HtestFout_all.mat
```

Validate the files and tensor shapes:

```bash
python read_dataset.py --data ./data/COST2100 --scenario in
# or
DATA=./data/COST2100 SCENARIO=out bash scripts/read_cost2100.sh
```

The loader expects sparse angular-delay CSI tensors shaped `(N, 2, 32, 32)` under key `HT`, and full-bandwidth complex CSI under key `HF_all` for the rho metric.

## Training

Train DCRNetV2-base on COST2100 outdoor with compression ratio 4:

```bash
python train.py \
  --variant base \
  --scenario out \
  --cr 4 \
  --data ./data/COST2100 \
  --gpu 0
```

Shell launcher equivalent:

```bash
DATA=./data/COST2100 VARIANT=base SCENARIO=out CR=4 GPU=0 bash scripts/train_cost2100.sh
```

Common options:

- `--variant {mini,small,base,unified,large-out,large-in4}`
- `--scenario {in,out}`
- `--cr {4,8,16,32}`
- `--epochs 1500 --batch-size 200 --lr 2e-3`
- `--outputs ./outputs`
- `--resume PATH` to resume optimizer/scheduler/model state
- `--finetune-from PATH --gate-reset -1.0` for fine-tuning hybrid variants

Checkpoints are written to `outputs/checkpoints/` and logs to `outputs/logs/*.jsonl`.

## Model variants

| Variant | Intended use |
|---|---|
| `mini` | smallest hybrid model for edge/mobile settings |
| `small` | low-compute direct low-rank decoder |
| `base` | balanced default |
| `unified` | one safer hybrid configuration across scenarios/CRs |
| `large-out` | outdoor-oriented hybrid configuration |
| `large-in4` | indoor cr=4-oriented hybrid configuration |

Example inference:

```python
import torch
from dcrnetv2_release import dcrnetv2_base

model = dcrnetv2_base(reduction=4)
ckpt = torch.load("outputs/checkpoints/DCRNetV2-base-out-cr4-best.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

x = torch.rand(1, 2, 32, 32)  # normalized angular-delay CSI
with torch.no_grad():
    y = model(x)
    z = model.encode(x)
```

## Citation

If this repository helps your work, please cite this repository and the original DCRNet paper, *Dilated Convolution Based CSI Feedback Compression for Massive MIMO Systems*.

```bibtex
@misc{dcrnetv2_repo,
  title={DCRNetV2: Lightweight CSI Feedback Compression Models},
  author={Tang, Shunpu},
  year={2026},
  howpublished={\url{https://github.com/tangshunpu/DCRNetV2}}
}
```

## License

This release is provided under the MIT License. See [LICENSE](LICENSE).
