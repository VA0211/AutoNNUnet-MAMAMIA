#!/bin/bash
# Waits for the folds 1-4 (300-epoch) orchestrator to finish, then resumes
# fold 0 from checkpoint_latest.pth (epoch ~358). Deliberately does NOT
# override hp_config.num_epochs/total_epochs, so it keeps its original
# 1000-epoch schedule consistent with the checkpoint it's resuming from
# (changing total_epochs mid-run would cause a LR schedule discontinuity).

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1
PARENT_PID=1941248

echo "$(date '+%F %T') fold-0 resume watcher (1000ep) started, PID $$, waiting on folds 1-4 orchestrator (PID ${PARENT_PID})" >> "$STATUS"

while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 30
done

rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_0"
echo "$(date '+%F %T') resuming fold 0 on GPU ${GPU} from checkpoint_latest.pth (original 1000-epoch schedule)" >> "$STATUS"
CUDA_VISIBLE_DEVICES=$GPU python runscripts/train.py \
  cluster=local \
  dataset=Dataset501_MAMAMIA \
  fold=0 \
  trainer.use_compressed_data=true \
  hydra.run.dir="$rundir" \
  > "$LOGDIR/fold_0_resume.log" 2>&1
rc=$?
echo "$(date '+%F %T') fold 0 resume finished (exit code ${rc})" >> "$STATUS"
echo "$(date '+%F %T') ALL 5 FOLDS COMPLETE (fold 0 @1000ep + folds 1-4 @300ep)" >> "$STATUS"
