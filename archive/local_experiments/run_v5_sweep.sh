#!/usr/bin/env bash
# Train DCRNet-v5 across r_enc x cr x scenario for a single ranks value.
#
# Usage:  ./run_v5_sweep.sh <ranks>
#
# ranks: one of {4, 8, 16}
# r_enc: 128 256 512
# CR:    4 8 16 32 64
# Scenario: in out
# Per ranks: 3 * 5 * 2 = 30 runs.
#
# Designed to be launched concurrently for different ranks values on the
# same GPU — v5 is tiny (~400K MACs), three streams share GPU 0 fine.
# Skips configs already marked OK in this rank's summary log so an
# interrupted stream can be resumed cleanly.

set -u

cd "$(dirname "$0")"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <ranks>" >&2
    exit 2
fi
ranks="$1"
case "$ranks" in
    4|8|16) ;;
    *) echo "ranks must be one of 4, 8, 16 (got '$ranks')" >&2; exit 2 ;;
esac

GPU=0
LR=2e-3
EPOCHS=1000
VAL_FREQ=10
MODEL=v5

R_ENCS=(128 256 512)
CRS=(4 8 16 32 64)
SCENARIOS=(in out)

SWEEP_ROOT="./outputs/v5_sweep"
mkdir -p "$SWEEP_ROOT"
SUMMARY="$SWEEP_ROOT/sweep_summary_r${ranks}.log"
echo "==== v5 sweep ranks=${ranks} started $(date -Iseconds) ====" | tee -a "$SUMMARY"

run_idx=0
total=$(( ${#R_ENCS[@]} * ${#CRS[@]} * ${#SCENARIOS[@]} ))

# Aggregate prior OK marks from both this stream's log and the original
# combined log so partially-completed work isn't redone.
prior_ok_files=("$SUMMARY")
[[ -f "$SWEEP_ROOT/sweep_summary.log" ]] && prior_ok_files+=("$SWEEP_ROOT/sweep_summary.log")

is_done() {
    local tag="$1"
    grep -F -- "OK   ${tag}" "${prior_ok_files[@]}" >/dev/null 2>&1
}

for r_enc in "${R_ENCS[@]}"; do
    out_dir="$SWEEP_ROOT/r${ranks}_renc${r_enc}"
    mkdir -p "$out_dir/log" "$out_dir/checkpoints"

    for cr in "${CRS[@]}"; do
        for scenario in "${SCENARIOS[@]}"; do
            run_idx=$((run_idx + 1))
            tag="ranks=${ranks} r_enc=${r_enc} cr=${cr} scenario=${scenario}"

            if is_done "$tag"; then
                echo "SKIP [${run_idx}/${total}] ${tag} (already OK)" | tee -a "$SUMMARY"
                continue
            fi

            echo
            echo "---- [${run_idx}/${total}] ${tag} ----" | tee -a "$SUMMARY"
            start=$(date +%s)

            python main.py \
                --model "$MODEL" \
                --gpu "$GPU" \
                --lr "$LR" \
                --epochs "$EPOCHS" \
                -v "$VAL_FREQ" \
                --cr "$cr" \
                --scenario "$scenario" \
                --expansion 1 \
                --ranks "$ranks" \
                --r-enc "$r_enc" \
                --outputs "$out_dir"
            status=$?

            elapsed=$(( $(date +%s) - start ))
            if [[ $status -eq 0 ]]; then
                echo "OK   ${tag} (${elapsed}s)" | tee -a "$SUMMARY"
            else
                echo "FAIL ${tag} status=${status} (${elapsed}s)" | tee -a "$SUMMARY"
            fi
        done
    done
done

echo "==== v5 sweep ranks=${ranks} finished $(date -Iseconds) ====" | tee -a "$SUMMARY"
