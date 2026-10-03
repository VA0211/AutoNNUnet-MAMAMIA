#!/bin/bash
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python
LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
GPU=1

echo "$(date '+%F %T') ResidualEncoderL single-fold (fold 1, best) test-set evaluation started, PID $$" >> "$STATUS"
CUDA_VISIBLE_DEVICES=$GPU taskset -c 0-39 "$PYTHON" run_test_evaluation_resencl_bestfold.py \
  > "$LOGDIR/test_evaluation_resencl_bestfold.log" 2>&1
rc=$?
echo "$(date '+%F %T') ResidualEncoderL single-fold test-set evaluation finished (exit code ${rc})" >> "$STATUS"
