#!/bin/bash
# Sequential per-bit QAT driver: runs 6→5→4→3→2 bits, 8 jobs in parallel per bit.
set -e
cd /home/ubuntu/Documents/project/DCRNet-V2
mkdir -p outputs/qat_logs
for BITS in 5 4 3 2; do
  echo "=== Starting bits=$BITS sweep ==="
  date
  for cr in 4 8 16 32; do
    for sc in in out; do
      ./.venv/bin/python qat_train.py --gpu 0 --scenario $sc --cr $cr --bits $BITS --epochs 50 \
        > outputs/qat_logs/qa${BITS}b_cr${cr}_${sc}.out 2>&1 &
    done
  done
  wait
  echo "=== Done bits=$BITS ==="
  date
done
echo "ALL bits sweep complete."
