#!/bin/bash
# 我（DSH agent）写的 mini-pilot：CPU 小规模，但评估口径与集中式基线一致（全量点）
set -u
cd /mnt/cfd/peiyu/CFD/FederatedLearning/repro/CFD
export MPLCONFIGDIR=/tmp/mpl-cfd OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python
COMMON="--data-root ../data --split easy --methods fedavg ppbc --alphas 0.1 \
  --rounds 6 --local-epochs 1 --downsample-size 3000 --test-downsample-size -1 \
  --max-train-samples 200 --max-test-samples 4 --batch-size 1 --eval-batch-size 1 \
  --num-workers 2 --device cpu --precision fp32 --eval-every 0 \
  --ppbc-iterations 2 --epoch-k 3 --iter-k 1"
echo "===== [1/2] single_class（每个 client 只含一种车身类型）====="
$PY experiments/ppbc_drivaernetpp_transformer_easy.py $COMMON \
  --num-clients 3 --client-split-mode single_class --single-class-allocation proportional \
  --output-dir ../results/minipilot_single_class
echo "===== [2/2] dirichlet alpha=0.1（极端异构）====="
$PY experiments/ppbc_drivaernetpp_transformer_easy.py $COMMON \
  --num-clients 3 --client-split-mode dirichlet \
  --output-dir ../results/minipilot_dirichlet
echo "[minipilot done]"
