# DCRNetV2

DCRNetV2 is a lightweight CSI feedback compression model family for FDD massive MIMO-OFDM systems. This repository publishes a self-contained PyTorch release with model definitions, COST2100 and RT-CSI dataset readers, and training scripts.

## Highlights

- **Complex Hermitian low-rank encoder**: learns matched-filter measurements while preserving real/imaginary cross terms.
- **Hybrid low-rank decoder**: combines a direct low-rank decoder with a gated factored residual bank for better high-compression reconstruction.
- **Ready-to-run variants**: DCRNetV2 (`mini`, `small`, `base`, `unified`, `large-out`, `large-in4`) and LRP (`lrp-r4-d128`, `lrp-r4-d512`, `lrp-r8-d512`, `lrp-r16-d512`).
- **COST2100 ready**: includes `.mat` readers, NMSE/rho evaluation, checkpointing, JSONL logs, and shell launchers.
- **RT-CSI ready**: reads normalized `.npz` splits for Mix5 scenes and separate Shenzhen adaptation/test archives; reports NMSE (full-bandwidth rho is unavailable).

## Figures

**DCRNetV2 architecture.** The overall CSI feedback path and its DCR encoder, fusion, factored residual decoder, and refinement blocks.

![DCRNetV2 architecture: overall flow and component diagrams](figures/DCRNetV2_archi.png)

**FLOPs versus NMSE.** The supplied comparison for compression ratios η=1/4 and η=1/16. Lower NMSE and fewer FLOPs are better.

![FLOPs versus NMSE comparison at compression ratios one quarter and one sixteenth](figures/flops_vs_nmse.png)

## Quick reproduction

1. Create an environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Put the COST2100 `.mat` files in `./data/COST2100` as shown [below](#dataset), then check their shapes:

   ```bash
   python read_dataset.py --data ./data/COST2100 --scenario in
   ```

3. Evaluate a curated checkpoint without training. This reads the two **test** files for the selected scenario; it prints NMSE in dB and rho:

   ```bash
   python evaluate.py --variant mini --scenario in --cr 4 --data ./data/COST2100
   ```

   The matching checkpoint is selected from `weights/dcrnetv2-mini/in/cr4.pt`. To evaluate a local checkpoint, add `--checkpoint PATH`. For a CPU run, add `--device cpu`. LRP uses the same command, for example `--variant lrp-r4-d128`. Keep the default batch size of 200 when comparing against paper NMSE; the paper protocol averages each batch's NMSE in dB. `--metric global` instead computes one NMSE over all test samples and can differ slightly.

   Check all 32 DCRNetV2 mini/small/base/unified paper NMSE values with one command (the test data for each scenario is loaded once):

   ```bash
   python scripts/verify_paper_results.py --data ./data/COST2100
   ```

   This checks the paper's two-decimal NMSE values to within 0.0051 dB. It does not recompute FLOPs, which depend on the model architecture rather than the checkpoint weights.

   Verification on the COST2100 test split (20,000 samples per scenario) matched **32/32** paper NMSE values; the largest absolute difference from the displayed two-decimal values was 0.0050 dB. This was run with Python 3.12, PyTorch 2.11, and a CUDA GPU.

4. To reproduce a training run, start with the small variant and then use the full recipe in [Training](#training):

   ```bash
   python train.py --variant mini --scenario in --cr 4 --data ./data/COST2100 \
     --epochs 1 --batch-size 2 --workers 0 --no-prefetch
   ```

   The one-epoch command checks the pipeline; it is not expected to match a pretrained model's result. Full training defaults to 1500 epochs. The current training entry selects the best checkpoint on validation NMSE and evaluates that checkpoint once on the held-out test split. The curated checkpoints come from earlier runs; see [weight provenance](weights/SOURCES.md).

## Repository layout

```text
.
├── models/
│   ├── dcrnetv2.py       # DCRNetV2 model definitions and variant factories
│   └── lrp.py            # LRP low-rank-prior variants
├── dataset/
│   ├── cost2100.py       # COST2100 .mat reader
│   └── rtcsi.py          # RT-CSI .npz reader
├── utils.py              # Scheduler, meters, NMSE/rho metrics
├── weights/              # Curated DCRNetV2/LRP checkpoints
├── figures/              # Paper figures used in README
├── archive/              # Legacy code kept out of the public path
├── train.py              # Single DCRNetV2/LRP training entry
├── evaluate.py           # Test-only evaluation of curated/local checkpoints
├── read_dataset.py       # Dataset validation/inspection entry
├── scripts/
│   ├── read_cost2100.sh  # Dataset smoke test launcher
│   ├── train_cost2100.sh # Training launcher
│   └── verify_paper_results.py # Check published NMSE values
├── requirements.txt
└── AGENTS.md
```

> Note: datasets, training outputs, and ad-hoc checkpoints are intentionally not tracked. Curated public checkpoints under `weights/` are tracked.

## Installation

```bash
git clone git@github.com:tangshunpu/DCRNetV2.git
cd DCRNetV2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch installation can vary by CUDA version. If needed, install the CUDA-specific wheel from the official PyTorch instructions first, then run `pip install -r requirements.txt`.

## Dataset

Download the COST2100 CSI-feedback benchmark used by CsiNet/DCRNet (see the [original DCRNet dataset instructions](https://github.com/tangshunpu/DCRNet#dataset)) and place the `.mat` files under `./data/COST2100` or pass a custom path with `--data`.

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

### RT-CSI NPZ data

Download the [RT-CSI release](https://github.com/tangshunpu/RT-CSI) and pass an `.npz` file directly with `--dataset rtcsi --data FILE`. The reader uses normalized `x_train`, `x_val`, and `x_test` arrays shaped `(N, 2, 32, 32)` in `[0, 1]`. Point `RTCSI_DATA` to the release's `data/` directory; the project does not embed a local dataset path.

```bash
export RTCSI_DATA=/path/to/RT-CSI-release/data
python read_dataset.py --dataset rtcsi --data "$RTCSI_DATA/csi_combined5_3GHz_32x1024.npz"
python train.py --dataset rtcsi --data "$RTCSI_DATA/csi_combined5_3GHz_32x1024.npz" \
  --variant unified --cr 4 --lr 1e-3
```

For Shenzhen adaptation, the few-shot training NPZ has no test rows. Supply the fixed test archive separately:

```bash
python read_dataset.py --dataset rtcsi \
  --data "$RTCSI_DATA/csi_SHENZHEN_train200_3GHz_32x1024.npz" \
  --test-data "$RTCSI_DATA/csi_SHENZHEN_test_3GHz_32x1024.npz"
```

No RT-CSI-trained weights are included in `weights/`. To measure **zero-shot transfer** from an existing COST2100 checkpoint, name it explicitly:

```bash
python evaluate.py --dataset rtcsi \
  --data "$RTCSI_DATA/csi_ZJU_3GHz_32x1024.npz" \
  --variant unified --cr 4 \
  --checkpoint weights/dcrnetv2-unified/out/cr4.pt
```

This prints NMSE only; RT-CSI NPZ files do not provide the `HF_all` tensor used for COST2100 rho. A checkpoint produced by `train.py --dataset rtcsi` can be evaluated with the same command by replacing `--checkpoint`.

Some older local RT-CSI experiment checkpoints use an earlier DCR encoder with fixed LeakyReLU slope 0.3 and old parameter names. They are **not bundled** with this release. If you have one, use `--checkpoint FILE --legacy-checkpoint` with the matching DCRNetV2 `--variant` and `--cr`; conversion is explicit and the resulting weights are strictly checked. For fine-tuning one of these files, use `train.py --finetune-from FILE --legacy-checkpoint` (not `--resume`).

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

Train an LRP configuration from the complexity-control table:

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

Checkpoints are written to `outputs/checkpoints/` and logs to `outputs/logs/*.jsonl`. Use `train.py` for the public DCRNetV2/LRP release; legacy DCRNet code is kept under `archive/`.

During training, the validation split determines the best checkpoint using validation NMSE. The selected checkpoint is then evaluated on the test split; COST2100 also reports rho. The JSONL log records the final test result in a row with `"phase": "test"`. To evaluate any checkpoint independently, run `evaluate.py --variant ... --cr ... --checkpoint PATH` with `--dataset` and the corresponding data options; optional evaluation flags are `--batch-size` (default 200), `--workers` (default 0), `--device {auto,cpu,cuda}` (default auto), and `--metric {paper,global}` (default paper).

## Model variants

| Variant | Intended use |
|---|---|
| `mini` | smallest hybrid DCRNetV2 model for edge/mobile settings |
| `small` | low-compute DCRNetV2 direct low-rank decoder |
| `base` | balanced DCRNetV2 default |
| `unified` | one safer DCRNetV2 hybrid configuration across scenarios/CRs |
| `large-out` | outdoor-oriented DCRNetV2 hybrid configuration |
| `large-in4` | indoor cr=4-oriented DCRNetV2 hybrid configuration |
| `lrp-r4-d128` | LRP with decoder rank `r=4`, encoder matched filters `d=128` |
| `lrp-r4-d512` | LRP with `r=4`, `d=512` |
| `lrp-r8-d512` | LRP with `r=8`, `d=512` |
| `lrp-r16-d512` | LRP with `r=16`, `d=512` |

### LRP complexity-control results

LRP is a low-rank-prior model. The table below reports the configurations used for model-based CSI feedback comparisons. FLOPs and NMSE are copied from the experiment table; NMSE is in dB.

| LRP config | eta=1/4 FLOPs | Indoor | Outdoor | eta=1/8 FLOPs | Indoor | Outdoor | eta=1/16 FLOPs | Indoor | Outdoor | eta=1/32 FLOPs | Indoor | Outdoor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lrp-r4-d128` | **0.68M** | -13.18 | -4.83 | **0.48M** | -12.19 | -4.32 | **0.39M** | -11.41 | -3.66 | **0.34M** | -8.58 | -2.40 |
| `lrp-r4-d512` | 1.88M | -14.11 | -5.96 | 1.49M | -14.15 | -5.30 | 1.29M | -11.82 | -4.00 | 1.20M | -8.61 | -2.64 |
| `lrp-r8-d512` | 2.16M | -17.94 | -8.33 | 1.64M | -15.81 | -6.14 | 1.38M | -12.47 | -4.19 | 1.25M | -8.62 | -2.70 |
| `lrp-r16-d512` | 2.72M | -21.62 | -9.04 | 1.93M | -15.87 | -6.30 | 1.54M | -12.55 | -4.19 | 1.34M | -8.67 | -2.72 |

Example inference with a curated DCRNetV2 checkpoint:

```python
import torch
from models.dcrnetv2 import dcrnetv2_large_out

ckpt = torch.load("weights/dcrnetv2-large-out/out/cr4.pt", map_location="cpu")
model = dcrnetv2_large_out(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

x = torch.rand(1, 2, 32, 32)  # normalized angular-delay CSI
with torch.no_grad():
    y = model(x)
    z = model.encode(x)
```

LRP can be loaded in the same way:

```python
import torch
from models.lrp import lrp_r16_d512

ckpt = torch.load("weights/lrp-r16-d512/in/cr8.pt", map_location="cpu")
model = lrp_r16_d512(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

## Pretrained weights

Curated checkpoints are stored under `weights/`:

```text
weights/
├── dcrnetv2-mini/{in,out}/cr{4,8,16,32}.pt
├── dcrnetv2-small/{in,out}/cr{4,8,16,32}.pt
├── dcrnetv2-base/{in,out}/cr{4,8,16,32}.pt
├── dcrnetv2-unified/{in,out}/cr{4,8,16,32}.pt
├── dcrnetv2-large-in4/in/cr{4,8,16,32}.pt
├── dcrnetv2-large-out/out/cr{4,8,16,32}.pt
├── lrp-r4-d128/{in,out}/cr{4,8,16,32}.pt
├── lrp-r4-d512/{in,out}/cr{4,8,16,32}.pt
├── lrp-r8-d512/{in,out}/cr{4,8,16,32}.pt
└── lrp-r16-d512/{in,out}/cr{4,8,16,32}.pt
```

Load a checkpoint with:

```python
import torch
from models.dcrnetv2 import dcrnetv2_large_out

model = dcrnetv2_large_out(reduction=4)
ckpt = torch.load("weights/dcrnetv2-large-out/out/cr4.pt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

LRP weights use the same format, e.g. `weights/lrp-r16-d512/in/cr8.pt`. Historical checkpoint-name mappings are documented in `weights/SOURCES.md`.

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
