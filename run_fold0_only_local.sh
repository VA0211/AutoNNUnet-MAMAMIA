#!/bin/bash
# Retrains fold 0 from scratch, 300 epochs, using the fully local venv.

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') fold-0 retrain (local venv, 300 epochs, from scratch) started, PID $$" >> "$STATUS"
echo "$(date '+%F %T') launching fold 0 on GPU ${GPU} (300 epochs, local venv)" >> "$STATUS"

rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_0"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=0 \
  trainer.use_compressed_data=true \
  hp_config.num_epochs=300 \
  hp_config.total_epochs=300 \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/fold_0.log" 2>&1
rc=$?
echo "$(date '+%F %T') fold 0 finished (exit code ${rc})" >> "$STATUS"
