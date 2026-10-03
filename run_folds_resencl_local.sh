#!/bin/bash
# Runs all 5 folds sequentially with the ResidualEncoderL backbone,
# 300 epochs each, on GPU 1, using the fully local venv. CPU hard-pinned
# to cores 0-39 via taskset, per server policy. Output goes to
# output/baseline_ResidualEncoderL/... (mirrors the ConvolutionalEncoder
# baseline layout, so the same evaluation tooling works unchanged).

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') ResidualEncoderL orchestrator (local venv, CPU-limited to cores 0-39) started, PID $$" >> "$STATUS"

for fold in 0 1 2 3 4; do
  rundir="output/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres/fold_${fold}"
  echo "$(date '+%F %T') launching ResidualEncoderL fold ${fold} on GPU ${GPU} (300 epochs, taskset 0-39, n_proc_DA=24)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-39 "$PYTHON" runscripts/train.py \
    cluster=local \
    dataset=Dataset501_MAMAMIA \
    fold=$fold \
    trainer.use_compressed_data=true \
    hp_config.encoder_type=ResidualEncoderL \
    hp_config.num_epochs=300 \
    hp_config.total_epochs=300 \
    hydra.job.env_set.nnUNet_n_proc_DA=24 \
    hydra.run.dir="$rundir" \
    > "$LOGDIR/resencl_fold_${fold}.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') ResidualEncoderL fold ${fold} finished (exit code ${rc})" >> "$STATUS"
done

echo "$(date '+%F %T') ResidualEncoderL orchestrator finished: all 5 folds complete" >> "$STATUS"
