# DCRNetV2

DCRNetV2 is a lightweight CSI feedback compression model family for FDD massive MIMO-OFDM systems. This repository publishes a self-contained PyTorch release with model definitions, COST2100 dataset reading utilities, and training scripts.

## Highlights

- **Complex Hermitian low-rank encoder**: learns matched-filter measurements while preserving real/imaginary cross terms.
- **Hybrid low-rank decoder**: combines a direct low-rank decoder with a gated factored residual bank for better high-compression reconstruction.
- **Ready-to-run variants**: DCRNetV2 (`mini`, `small`, `base`, `unified`, `large-out`, `large-in4`) and LRP (`lrp-r4-d128`, `lrp-r4-d512`, `lrp-r8-d512`, `lrp-r16-d512`).
- **COST2100 ready**: includes `.mat` readers, NMSE/rho evaluation, checkpointing, JSONL logs, and shell launchers.

## Repository layout

```text
.
├── models/
│   ├── dcrnetv2.py       # DCRNetV2 model definitions and variant factories
│   └── lrp.py            # LRP (renamed v5) low-rank-prior variants
├── dataset/
│   └── cost2100.py       # COST2100 loading + validation CLI
├── utils.py              # Scheduler, meters, NMSE/rho metrics
├── train.py              # Single DCRNetV2 training entry
├── read_dataset.py       # Dataset validation/inspection entry
├── scripts/
│   ├── read_cost2100.sh  # Dataset smoke test launcher
│   └── train_cost2100.sh # Training launcher
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

Train an LRP/v5 configuration from the complexity-control table:

```bash
python train.py --variant lrp-r16-d512 --scenario out --cr 4 --data ./data/COST2100 --gpu 0
```

Common options:

- `--variant {mini,small,base,unified,large-out,large-in4,lrp-r4-d128,lrp-r4-d512,lrp-r8-d512,lrp-r16-d512}`
- `--scenario {in,out}`
- `--cr {4,8,16,32}`
- `--epochs 1500 --batch-size 200 --lr 2e-3`
- `--outputs ./outputs`
- `--resume PATH` to resume optimizer/scheduler/model state
- `--finetune-from PATH --gate-reset -1.0` for fine-tuning hybrid variants

Checkpoints are written to `outputs/checkpoints/` and logs to `outputs/logs/*.jsonl`. `main.py` is kept only for legacy/experimental DCRNet-V1 workflows; use `train.py` for the public DCRNetV2 release.

## Model variants

| Variant | Intended use |
|---|---|
| `mini` | smallest hybrid DCRNetV2 model for edge/mobile settings |
| `small` | low-compute DCRNetV2 direct low-rank decoder |
| `base` | balanced DCRNetV2 default |
| `unified` | one safer DCRNetV2 hybrid configuration across scenarios/CRs |
| `large-out` | outdoor-oriented DCRNetV2 hybrid configuration |
| `large-in4` | indoor cr=4-oriented DCRNetV2 hybrid configuration |
| `lrp-r4-d128` | LRP/v5 with decoder rank `r=4`, encoder matched filters `d=128` |
| `lrp-r4-d512` | LRP/v5 with `r=4`, `d=512` |
| `lrp-r8-d512` | LRP/v5 with `r=8`, `d=512` |
| `lrp-r16-d512` | LRP/v5 with `r=16`, `d=512` |

### LRP complexity-control results

LRP is the renamed v5 low-rank-prior model. The table below reports the configurations used for model-based CSI feedback comparisons. FLOPs and NMSE are copied from the experiment table; NMSE is in dB.

| LRP config | eta=1/4 FLOPs | Indoor | Outdoor | eta=1/8 FLOPs | Indoor | Outdoor | eta=1/16 FLOPs | Indoor | Outdoor | eta=1/32 FLOPs | Indoor | Outdoor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lrp-r4-d128` | **0.68M** | -13.18 | -4.83 | **0.48M** | -12.19 | -4.32 | **0.39M** | -11.41 | -3.66 | **0.34M** | -8.58 | -2.40 |
| `lrp-r4-d512` | 1.88M | -14.11 | -5.96 | 1.49M | -14.15 | -5.30 | 1.29M | -11.82 | -4.00 | 1.20M | -8.61 | -2.64 |
| `lrp-r8-d512` | 2.16M | -17.94 | -8.33 | 1.64M | -15.81 | -6.14 | 1.38M | -12.47 | -4.19 | 1.25M | -8.62 | -2.70 |
| `lrp-r16-d512` | 2.72M | -21.62 | -9.04 | 1.93M | -15.87 | -6.30 | 1.54M | -12.55 | -4.19 | 1.34M | -8.67 | -2.72 |

Example inference:

```python
import torch
from models.dcrnetv2 import dcrnetv2_base

model = dcrnetv2_base(reduction=4)
ckpt = torch.load("outputs/checkpoints/DCRNetV2-base-out-cr4-best.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

x = torch.rand(1, 2, 32, 32)  # normalized angular-delay CSI
with torch.no_grad():
    y = model(x)
    z = model.encode(x)
```

LRP can be loaded in the same way:

```python
from models.lrp import lrp_r16_d512
model = lrp_r16_d512(reduction=4)
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
