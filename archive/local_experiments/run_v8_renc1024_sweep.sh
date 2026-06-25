#!/usr/bin/env bash
# v8 1X r=16 r_enc=1024 sweep — indoor cr=4/8/16/32 parallel, then outdoor.
# 1000 ep each, batch=200, lr=1e-3, warmup=15, no EMA.
set -u

cd /home/ubuntu/Documents/project/DCRNet-V2
source .venv/bin/activate

CRS=(4 8 16 32)
COMMON_ARGS=(
  --model v8 --gpu 0
  --expansion 1 --ranks 16 --r-enc 1024 --no-ema
  -b 200 --lr 1e-3 --warmup-epochs 15 --epochs 1000
)

run_phase () {
  local scenario=$1
  echo "[$(date +%H:%M:%S)] Phase: scenario=${scenario}  cr=${CRS[*]}"
  local pids=()
  for cr in "${CRS[@]}"; do
    local out="outputs/v8new_renc1024_cr${cr}_${scenario}.out"
    python train_modern.py "${COMMON_ARGS[@]}" \
      --scenario "${scenario}" --cr "${cr}" \
      > "${out}" 2>&1 &
    local pid=$!
    pids+=("${pid}")
    echo "  scenario=${scenario} cr=${cr} PID=${pid}  log=${out}"
  done
  # Wait for *all* phase pids — if any crashes its 'wait' returns nonzero
  # but we still wait for the rest.
  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      echo "  PID=${pid} exited nonzero" >&2
      rc=1
    fi
  done
  echo "[$(date +%H:%M:%S)] Phase ${scenario} done (rc=${rc})"
  return ${rc}
}

run_phase in
run_phase out
echo "[$(date +%H:%M:%S)] all done"
