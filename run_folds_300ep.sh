#!/bin/bash
# Runs folds 1-4 sequentially on a single GPU with a 300-epoch budget
# (num_epochs=total_epochs=300, so the LR schedule anneals properly over
# the shorter horizon). Fold 0 keeps its original 1000-epoch schedule
# and is resumed separately (see resume_fold0_1000ep.sh).

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') 300-epoch orchestrator started, PID $$ (folds 1-4 on GPU ${GPU}, num_epochs=total_epochs=300)" >> "$STATUS"

for fold in 1 2 3 4; do
  rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_${fold}"
  echo "$(date '+%F %T') launching fold ${fold} on GPU ${GPU} (300 epochs)" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$GPU python runscripts/train.py \
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
