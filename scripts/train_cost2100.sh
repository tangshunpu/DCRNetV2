#!/usr/bin/env bash
set -euo pipefail

# Example training launcher for DCRNetV2 on COST2100.
# Override values from the environment, e.g.:
#   DATA=./data/COST2100 VARIANT=base SCENARIO=out CR=4 GPU=0 bash scripts/train_cost2100.sh

DATA=${DATA:-./data/COST2100}
VARIANT=${VARIANT:-base}
SCENARIO=${SCENARIO:-out}
CR=${CR:-4}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-1500}
BATCH_SIZE=${BATCH_SIZE:-200}
LR=${LR:-0.002}
OUTPUTS=${OUTPUTS:-./outputs}

python train.py \
  --data "${DATA}" \
  --variant "${VARIANT}" \
  --scenario "${SCENARIO}" \
  --cr "${CR}" \
  --gpu "${GPU}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --outputs "${OUTPUTS}"
