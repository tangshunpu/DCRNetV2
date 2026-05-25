#!/usr/bin/env bash
set -euo pipefail

# Validate COST2100 files and print tensor/DataLoader shapes.
DATA=${DATA:-./data/COST2100}
SCENARIO=${SCENARIO:-in}

python read_dataset.py --data "${DATA}" --scenario "${SCENARIO}" --batch-size 4 --workers 0
