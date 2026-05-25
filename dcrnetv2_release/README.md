# DCRNetV2 release package

This directory contains the self-contained public DCRNetV2 implementation:

- `dcrnetv2.py`: model definitions and variant factory functions
- `dataset.py`: COST2100 `.mat` reader and validation CLI
- `train.py`: training/checkpoint/logging entry
- `utils.py`: scheduler, meters, NMSE/rho metrics

Recommended commands from the repository root:

```bash
python read_dataset.py --data ./data/COST2100 --scenario in
python train.py --variant base --scenario out --cr 4 --data ./data/COST2100
```

You can also run this package directly:

```bash
python -m dcrnetv2_release.dataset --data ./data/COST2100 --scenario out
python -m dcrnetv2_release.train --variant mini --scenario in --cr 16
```

See the root `README.md` for full setup, dataset layout, training recipes, and license.
