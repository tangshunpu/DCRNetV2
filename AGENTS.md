# AGENTS.md

Guidance for AI/code agents working in this repository.

## Scope

The public release lives in `dcrnetv2_release/` plus the root wrappers `train.py` and `read_dataset.py`. Prefer editing these files for release-facing changes.

## Do not commit

- COST2100 or any other datasets (`data/`, `COST2100`, `*.mat`).
- Training outputs/checkpoints/logs (`outputs/`, `runs/`, `wandb/`, `*.pt`, `*.pth`, `*.ckpt`).
- Local virtual environments, caches, or machine-specific paths.

## Validation checklist

Before committing release changes, run:

```bash
python -m py_compile train.py read_dataset.py dcrnetv2_release/*.py
python - <<'PY'
import torch
from dcrnetv2_release import dcrnetv2_mini
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
- Use relative imports inside `dcrnetv2_release` so both `python train.py` and `python -m dcrnetv2_release.train` work.
- Document new CLI flags in `README.md`.
