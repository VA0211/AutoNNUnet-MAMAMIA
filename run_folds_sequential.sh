#!/bin/bash
# Sequential single-GPU runner: waits for the already-running fold 0 to finish,
# then runs folds 1-4 one at a time on a single GPU, to minimize footprint on
# shared lab GPUs. Intended to be launched via nohup + disown.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1
FOLD0_PID=4009352

echo "$(date '+%F %T') sequential orchestrator started, PID $$, waiting on existing fold 0 (PID ${FOLD0_PID}) on GPU ${GPU}" >> "$STATUS"

while kill -0 "$FOLD0_PID" 2>/dev/null; do
  sleep 30
done
echo "$(date '+%F %T') fold 0 finished" >> "$STATUS"

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

echo "$(date '+%F %T') sequential orchestrator finished: all 5 folds complete" >> "$STATUS"
