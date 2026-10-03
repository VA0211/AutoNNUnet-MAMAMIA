#!/bin/bash
# Runs ResidualEncoderL fold 1 only, 300 epochs, GPU 1, CPU pinned 0-39.

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') ResidualEncoderL fold-1-only (local venv, CPU-limited) started, PID $$" >> "$STATUS"
echo "$(date '+%F %T') launching ResidualEncoderL fold 1 on GPU ${GPU} (300 epochs, taskset 0-39, n_proc_DA=24)" >> "$STATUS"

rundir="output/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres/fold_1"
CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-39 "$PYTHON" runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=1 \
  trainer.use_compressed_data=true \
  hp_config.encoder_type=ResidualEncoderL \
  hp_config.num_epochs=300 \
  hp_config.total_epochs=300 \
  hydra.job.env_set.nnUNet_n_proc_DA=24 \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/resencl_fold_1.log" 2>&1
rc=$?
echo "$(date '+%F %T') ResidualEncoderL fold 1 finished (exit code ${rc})" >> "$STATUS"
