#!/bin/bash
# Waits for the currently-running folds 1-4 orchestrator to finish, then
# resumes fold 0 (crashed at epoch ~358/1000 due to disk-full) from its
# checkpoint_latest.pth via AutoNNUnet's default continue_training=true.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1
PARENT_PID=1836536

echo "$(date '+%F %T') fold-0 resume watcher started, PID $$, waiting on folds 1-4 orchestrator (PID ${PARENT_PID})" >> "$STATUS"

while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 30
done

rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_0"
echo "$(date '+%F %T') resuming fold 0 on GPU ${GPU} from checkpoint_latest.pth" >> "$STATUS"
CUDA_VISIBLE_DEVICES=$GPU python runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=0 \
  trainer.use_compressed_data=true \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/fold_0_resume.log" 2>&1
rc=$?
echo "$(date '+%F %T') fold 0 resume finished (exit code ${rc})" >> "$STATUS"
echo "$(date '+%F %T') ALL 5 FOLDS COMPLETE (fold 0 resumed + folds 1-4)" >> "$STATUS"
