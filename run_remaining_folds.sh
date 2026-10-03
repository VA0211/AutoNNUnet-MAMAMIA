#!/bin/bash
# Relaunches folds 1-4 sequentially on a single GPU, after they crashed
# earlier due to disk being completely full. Fold 0 already completed
# successfully and is left untouched. Intended for nohup + disown.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') relaunch orchestrator started, PID $$ (folds 1-4 on GPU ${GPU}, fold 0 already complete)" >> "$STATUS"

for fold in 1 2 3 4; do
  rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_${fold}"
  echo "$(date '+%F %T') launching fold ${fold} on GPU ${GPU}" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$GPU python runscripts/train.py \
    cluster=local \
    dataset=Dataset501_MAMAMIA \
    fold=$fold \
    trainer.use_compressed_data=true \
    hydra.run.dir="$rundir" \
    > "$LOGDIR/fold_${fold}.log" 2>&1
  rc=$?
  echo "$(date '+%F %T') fold ${fold} finished (exit code ${rc})" >> "$STATUS"
done

echo "$(date '+%F %T') relaunch orchestrator finished: all 5 folds complete" >> "$STATUS"
