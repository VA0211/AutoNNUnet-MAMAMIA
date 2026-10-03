#!/bin/bash
# Runs fold 4 only on GPU 1, 300 epochs, using the fully local venv.
# CPU hard-pinned to cores 0-39 via taskset, per server policy.

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') fold-4-only orchestrator (local venv, CPU-limited to cores 0-39) started, PID $$" >> "$STATUS"
echo "$(date '+%F %T') launching fold 4 on GPU ${GPU} (300 epochs, local venv, taskset 0-39, n_proc_DA=24)" >> "$STATUS"

rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_4"
CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-39 "$PYTHON" runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=4 \
  trainer.use_compressed_data=true \
  hp_config.num_epochs=300 \
  hp_config.total_epochs=300 \
  hydra.job.env_set.nnUNet_n_proc_DA=24 \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/fold_4.log" 2>&1
rc=$?
echo "$(date '+%F %T') fold 4 finished (exit code ${rc})" >> "$STATUS"
