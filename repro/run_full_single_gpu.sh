#!/usr/bin/env bash
# Compute-matched full-participation FedAvg reference and PPBC, on ONE GPU.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON=${PYTHON:-python}
if [[ "$PYTHON" == */* ]]; then PYTHON=$(realpath -m -- "$PYTHON"); fi
DATA_ROOT=$(realpath -m -- "${DATA_ROOT:-$ROOT/data}")
STUDY_DIR=$(realpath -m -- "${STUDY_DIR:-$ROOT/results/full_single_gpu_$(date -u +%Y%m%dT%H%M%S)}")
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-cfd}
export PYTHONUNBUFFERED=1
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'Select exactly one GPU.' >&2; exit 2
fi
if [[ -e "$STUDY_DIR" && "${RESUME:-0}" != 1 ]]; then
  echo "Existing study requires RESUME=1: $STUDY_DIR" >&2; exit 2
fi
mkdir -p "$STUDY_DIR"
exec 9>"$STUDY_DIR/study.lock"
flock -n 9 || { echo 'This study already has a running process.' >&2; exit 2; }
printf '%s\n' "$$" > "$STUDY_DIR/runner.pid"
if [[ ! -f "$STUDY_DIR/source_commit.txt" ]]; then
  git -C "$ROOT/.." rev-parse HEAD > "$STUDY_DIR/source_commit.txt"
fi
echo "Study: $STUDY_DIR; physical GPU: $CUDA_VISIBLE_DEVICES"
cd "$ROOT/CFD"
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required'
x = torch.ones(32, 32, device='cuda')
assert (x @ x).sum().item() == 32768
print(torch.cuda.get_device_name(), torch.__version__, flush=True)
PY
COMMON=(--data-root "$DATA_ROOT" --split easy --alphas 0.1 --seed 42
  --num-clients 6 --client-split-mode single_class --single-class-allocation equal_classes
  --aggregation equal --local-epochs 1 --batch-size 1 --eval-batch-size 1
  --downsample-size 5000 --test-downsample-size -1 --num-workers 2
  --device cuda --precision fp32 --lr 1e-4 --weight-decay 0.01
  --ppbc-iterations 2 --epoch-k 6 --iter-k 2 --theta 0.25 --gamma 1 --q-m 1)
for METHOD in ppbc fedavg; do
  if [[ "$METHOD" == ppbc ]]; then ROUNDS=30; EVAL=5; else ROUNDS=60; EVAL=10; fi
  DEST="$STUDY_DIR/$METHOD"
  CMD=("$PYTHON" experiments/ppbc_drivaernetpp_transformer_easy.py "${COMMON[@]}"
    --methods "$METHOD" --rounds "$ROUNDS" --eval-every "$EVAL" --output-dir "$DEST")
  CHECKPOINT="$DEST/latest_${METHOD}_alpha_0.1.pt"
  if [[ "${RESUME:-0}" == 1 && -f "$CHECKPOINT" ]]; then CMD+=(--resume "$CHECKPOINT"); fi
  printf '%q ' "${CMD[@]}" > "$STUDY_DIR/${METHOD}_command.sh"
  printf '\n' >> "$STUDY_DIR/${METHOD}_command.sh"
  printf '%s\n' "running $METHOD $(date -u +%FT%TZ)" > "$STUDY_DIR/status.txt"
  if "${CMD[@]}"; then
    printf '%s\n' "$METHOD complete $(date -u +%FT%TZ)" >> "$STUDY_DIR/completed.txt"
  else
    STATUS=$?
    printf '%s\n' "failed $METHOD exit=$STATUS $(date -u +%FT%TZ)" > "$STUDY_DIR/status.txt"
    exit "$STATUS"
  fi
done
printf '%s\n' "complete $(date -u +%FT%TZ)" > "$STUDY_DIR/status.txt"
