#!/usr/bin/env bash
# Run the existing CFD PPBC adaptation on one visible GPU.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
MODE=${1:-smoke}
case "$MODE" in
  smoke) ROUNDS=2; TRAIN=18; TEST=3; POINTS=1000; BATCH_CAP=1 ;;
  pilot) ROUNDS=6; TRAIN=200; TEST=12; POINTS=3000; BATCH_CAP=0 ;;
  *) echo 'Usage: [GPU=0 PYTHON=/path/to/python DATA_ROOT=/path] bash run_single_gpu.sh [smoke|pilot]' >&2; exit 2 ;;
esac
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/results/single_gpu_${MODE}_$(date -u +%Y%m%dT%H%M%S%N)}
export CUDA_VISIBLE_DEVICES=${GPU:-${CUDA_VISIBLE_DEVICES:-0}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-cfd}
PRECISION=${PRECISION:-fp32}
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'Select exactly one GPU using GPU or CUDA_VISIBLE_DEVICES.' >&2; exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2; exit 2
fi
# Resolve caller-supplied relative paths before switching to the CFD directory.
DATA_ROOT=$(realpath -m -- "$DATA_ROOT")
OUTPUT_DIR=$(realpath -m -- "$OUTPUT_DIR")
if [[ "$PYTHON" == */* ]]; then PYTHON=$(realpath -m -- "$PYTHON"); fi
CMD=("$PYTHON" experiments/ppbc_drivaernetpp_transformer_easy.py
  --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR"
  --split easy --methods ppbc --alphas 0.1 --seed 42
  --num-clients 3 --client-split-mode single_class
  --single-class-allocation equal_classes --aggregation equal
  --rounds "$ROUNDS" --local-epochs 1 --ppbc-iterations 2
  --epoch-k 3 --iter-k 1 --theta 0.25 --gamma 1 --q-m 1
  --max-train-samples "$TRAIN" --max-test-samples "$TEST"
  --max-client-batches "$BATCH_CAP" --downsample-size "$POINTS"
  --test-downsample-size -1 --batch-size 1 --eval-batch-size 1
  --num-workers 0 --device cuda --precision "$PRECISION" --eval-every 1)
cd "$ROOT/CFD"
printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
"$PYTHON" - "$DATA_ROOT" "$PRECISION" <<'PY'
import json
import sys
from pathlib import Path
import torch
from model.transformer import Transformer
from util.cfd_utils import calc_force_coeff_wrapper

root = Path(sys.argv[1])
for name in ['mean_std.json', 'split/easy/train_split.json', 'split/easy/test_split.json']:
    with (root / name).open() as f:
        json.load(f)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('A single working CUDA GPU is required; no CPU fallback was started.')
if sys.argv[2] == 'bf16' and not torch.cuda.is_bf16_supported():
    raise SystemExit('Selected GPU does not support bf16; use PRECISION=fp32.')
x = torch.ones((32, 32), device='cuda')
assert (x @ x).sum().item() == 32768
props = torch.cuda.get_device_properties(0)
print(f'GPU={props.name}; VRAM={props.total_memory / 2**30:.1f} GiB; '
      f'torch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
PY
exec "${CMD[@]}"
