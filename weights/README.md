# Pretrained weights

Curated checkpoints for COST2100 and Sionna-RT-Mix5. The RT-CSI checkpoints use the current DCRNetV2 model format and load strictly without conversion flags.

## Layout

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
├── lrp-r16-d512/{in,out}/cr{4,8,16,32}.pt
└── rtcsi-mix5/dcrnetv2-unified/cr{4,8,16}.pt
```

Each file is a PyTorch dictionary with at least:

- `state_dict`
- `model_family`
- `variant`
- `cr`
- `best_nmse`
- `best_epoch`

COST2100 checkpoints also record `scenario` (`in` or `out`); RT-CSI checkpoints record `dataset=rtcsi` and `training_data=Sionna-RT-Mix5`.

## Loading

```python
import torch
from models.dcrnetv2 import dcrnetv2_large_out

ckpt = torch.load("weights/dcrnetv2-large-out/out/cr4.pt", map_location="cpu")
model = dcrnetv2_large_out(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

For RT-CSI:

```python
import torch
from models.dcrnetv2 import dcrnetv2_unified

ckpt = torch.load("weights/rtcsi-mix5/dcrnetv2-unified/cr4.pt", map_location="cpu")
model = dcrnetv2_unified(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
```

For LRP:

```python
import torch
from models.lrp import lrp_r16_d512

ckpt = torch.load("weights/lrp-r16-d512/in/cr8.pt", map_location="cpu")
model = lrp_r16_d512(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

See `SOURCES.md` for the historical checkpoint names used to build each curated file.
