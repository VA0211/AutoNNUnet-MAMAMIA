#!/bin/bash
# Runs fold 1 only on GPU 1, using the fully local venv.

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') fold-1-only orchestrator (local venv) started, PID $$" >> "$STATUS"
echo "$(date '+%F %T') launching fold 1 on GPU ${GPU} (300 epochs, local venv)" >> "$STATUS"

rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_1"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=1 \
  trainer.use_compressed_data=true \
  hp_config.num_epochs=300 \
  hp_config.total_epochs=300 \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/fold_1.log" 2>&1
rc=$?
echo "$(date '+%F %T') fold 1 finished (exit code ${rc})" >> "$STATUS"
