# Pretrained weights

Curated checkpoints for the public DCRNetV2/LRP release.

These checkpoints were trained on COST2100. There are no RT-CSI-trained checkpoints in this directory; using one on RT-CSI data is a zero-shot transfer test and requires an explicit `--checkpoint` argument.
Older local RT-CSI checkpoints may exist in ignored experiment outputs. Use `evaluate.py --legacy-checkpoint --checkpoint FILE` to convert a matching DCRNetV2 variant at load time; these files are not part of the public weight set.

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
└── lrp-r16-d512/{in,out}/cr{4,8,16,32}.pt
```

Each file is a PyTorch dictionary with at least:

- `state_dict`
- `model_family`
- `variant`
- `scenario`
- `cr`
- `best_nmse`
- `best_epoch`

## Loading

```python
import torch
from models.dcrnetv2 import dcrnetv2_large_out

ckpt = torch.load("weights/dcrnetv2-large-out/out/cr4.pt", map_location="cpu")
model = dcrnetv2_large_out(reduction=ckpt["cr"])
model.load_state_dict(ckpt["state_dict"])
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
