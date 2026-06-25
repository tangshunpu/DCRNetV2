#!/bin/bash
set -e
cd /home/ubuntu/Documents/project/DCRNet-V2
for BPD in 6.0 5.0 4.0 3.0 2.0; do
  echo "=== bpd=$BPD batch ==="; date
  for cr in 4 8 16 32; do
    for sc in in out; do
      ./.venv/bin/python fixed_rate_train.py --gpu 0 --scenario $sc --cr $cr \
        --target-bpd $BPD --epochs 50 --lr 5e-5 \
        > outputs/qat_logs/fbpd${BPD}_cr${cr}_${sc}.out 2>&1 &
    done
  done
  wait
  echo "=== bpd=$BPD done ==="; date
done
echo "ALL FIXED-RATE SWEEP COMPLETE."
