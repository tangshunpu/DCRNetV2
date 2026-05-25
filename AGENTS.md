# AGENTS.md

Guidance for AI/code agents working in this repository.

## Scope

The public DCRNetV2 release is integrated into the normal project layout:

- `models/dcrnetv2.py` for the model.
- `dataset/cost2100.py` for COST2100 reading/validation.
- `utils.py` for training metrics/scheduler helpers.
- `train.py` as the only public DCRNetV2 training entry.
- `read_dataset.py` for dataset smoke tests.

Do not recreate a separate `dcrnetv2_release/` package unless explicitly requested.

## Do not commit

- COST2100 or any other datasets (`data/`, `COST2100`, `*.mat`).
- Training outputs/checkpoints/logs (`outputs/`, `runs/`, `wandb/`, `*.pt`, `*.pth`, `*.ckpt`).
- Local virtual environments, caches, or machine-specific paths.

## Validation checklist

Before committing release changes, run:

```bash
python -m py_compile train.py read_dataset.py utils.py models/dcrnetv2.py dataset/cost2100.py
python - <<'PY'
import torch
from models.dcrnetv2 import dcrnetv2_mini
m = dcrnetv2_mini(reduction=4).eval()
x = torch.rand(1, 2, 32, 32)
with torch.no_grad():
    y = m(x)
print(tuple(y.shape))
PY
```

If a small synthetic COST2100 fixture is available, also smoke-test one epoch with `--variant mini --epochs 1 --batch-size 2 --workers 0 --no-prefetch`.

## Style

- Keep the release scripts self-contained and easy to run from a fresh clone.
- Avoid duplicate model/training packages; put release code in the existing `models/`, `dataset/`, and root script locations.
- Document new CLI flags in `README.md`.
