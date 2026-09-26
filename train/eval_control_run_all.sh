#!/usr/bin/env bash
# Evidence-prompt control run: re-evaluates the three Format-A stage-2 adapters
# under the evidence prompt, isolating prompt format from training target.
# Run inside an srun window -- do NOT set CUDA_VISIBLE_DEVICES, srun allocates it.
set -uo pipefail

PY=python3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN=$REPO/train
CTL=$REPO/eval_control
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
echo "start $(date)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

for RUN in stage2_from_base stage2_from_frozen stage2_from_train_vision_evi; do
    if [ -f "$CTL/$RUN/benchmark_predictions.jsonl" ]; then
        echo "=== $RUN  already done, skipping ==="; continue
    fi
    echo "=== $RUN  $(date +%H:%M:%S) ==="
    "$PY" -u "$TRAIN/eval_stage2.py" \
        --model    "$REPO/models/Qwen3.5-2B" \
        --adapter  "$REPO/runs/$RUN/final" \
        --data-dir "$CTL/bench_local" \
        --split    benchmark \
        --target-format evidence_answer \
        --gen-n -1 --loss-n 0 --gen-batch-size 16 \
        --out-dir  "$CTL/$RUN" 2>&1 | tee "$CTL/$RUN.log"
done

echo; echo "=== summary ==="
(cd "$CTL" && "$PY" "$TRAIN/eval_control_summarize.py")
echo "done $(date)"
