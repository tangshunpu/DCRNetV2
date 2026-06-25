#!/bin/bash
set -e
cd /home/ubuntu/Documents/project/DCRNet-V2
mkdir -p outputs/qat_logs
for LAM in 1e-8 1e-7 1e-6 1e-5; do
  echo "=== λ=$LAM batch ==="; date
  for cr in 4 8 16 32; do
    for sc in in out; do
      ./.venv/bin/python entropy_train.py --gpu 0 --scenario $sc --cr $cr \
        --lam $LAM --epochs 30 --prior-lr 5e-4 --freeze-model \
        > outputs/qat_logs/ent_lam${LAM}_cr${cr}_${sc}.out 2>&1 &
    done
  done
  wait
  echo "=== λ=$LAM done ==="; date
done
echo "ALL ENTROPY SWEEP COMPLETE."
