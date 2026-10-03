#!/bin/bash
# Orchestrates a full 5-fold AutoNNUnet training run on Dataset501_MAMAMIA,
# using only the GPUs that were idle at launch time, queueing extra folds
# onto GPUs as they free up. Fully self-contained: intended to be launched
# via nohup + disown so it survives the launching shell/session ending.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status.log"

echo "$(date '+%F %T') orchestrator started, PID $$" >> "$STATUS"

IDLE_GPUS=(1 3 5)
ALL_FOLDS=(0 1 2 3 4)
pending=("${ALL_FOLDS[@]}")
declare -A running_pid_for_gpu   # gpu -> pid
declare -A fold_for_gpu          # gpu -> fold

run_fold_on_gpu() {
  local fold=$1
  local gpu=$2
  local rundir="output/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres/fold_${fold}"
  echo "$(date '+%F %T') launching fold ${fold} on GPU ${gpu}" >> "$STATUS"
  CUDA_VISIBLE_DEVICES=$gpu nohup python runscripts/train.py \
    cluster=local \
    dataset=Dataset501_MAMAMIA \
    fold=$fold \
    trainer.use_compressed_data=true \
    hydra.run.dir="$rundir" \
    > "$LOGDIR/fold_${fold}.log" 2>&1 &
  running_pid_for_gpu[$gpu]=$!
  fold_for_gpu[$gpu]=$fold
}

# Kick off the first wave on all idle GPUs
for gpu in "${IDLE_GPUS[@]}"; do
  if [ ${#pending[@]} -eq 0 ]; then break; fi
  fold=${pending[0]}
  pending=("${pending[@]:1}")
  run_fold_on_gpu "$fold" "$gpu"
done

# Queue manager: poll running jobs, backfill pending folds as GPUs free up
while true; do
  any_running=false
  for gpu in "${!running_pid_for_gpu[@]}"; do
    pid=${running_pid_for_gpu[$gpu]}
    if kill -0 "$pid" 2>/dev/null; then
      any_running=true
    else
      f=${fold_for_gpu[$gpu]}
      wait "$pid"
      rc=$?
      echo "$(date '+%F %T') fold ${f} on GPU ${gpu} finished (exit code ${rc})" >> "$STATUS"
      unset running_pid_for_gpu[$gpu]
      unset fold_for_gpu[$gpu]
      if [ ${#pending[@]} -gt 0 ]; then
        next=${pending[0]}
        pending=("${pending[@]:1}")
        run_fold_on_gpu "$next" "$gpu"
        any_running=true
      fi
    fi
  done
  if [ "$any_running" = false ] && [ ${#pending[@]} -eq 0 ]; then
    break
  fi
  sleep 60
done

echo "$(date '+%F %T') orchestrator finished: all 5 folds complete" >> "$STATUS"
