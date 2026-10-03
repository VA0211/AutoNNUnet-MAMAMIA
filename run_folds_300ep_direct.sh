#!/bin/bash
# Same as run_folds_300ep.sh but invokes the env's python binary directly
# instead of `conda activate`, which touches many activation-hook files
# and has been a hang point during the recent NFS instability.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/workspace/hailt/miniconda3/envs/autonnunet/bin/python

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') 300-epoch orchestrator (direct python, no conda activate) started, PID $$" >> "$STATUS"

for fold in 1 2 3 4; do
  rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_${fold}"
  echo "$(date '+%F %T') launching fold ${fold} on GPU ${GPU} (300 epochs, direct python)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" runscripts/train.py \
    cluster=local \
    dataset=Dataset501_MAMAMIA \
    fold=$fold \
    trainer.use_compressed_data=true \
    hp_config.num_epochs=300 \
    hp_config.total_epochs=300 \
    hydra.run.dir="$rundir" \
    > "$LOGDIR/fold_${fold}.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') fold ${fold} finished (exit code ${rc})" >> "$STATUS"
done

echo "$(date '+%F %T') 300-epoch orchestrator finished: folds 1-4 complete" >> "$STATUS"
