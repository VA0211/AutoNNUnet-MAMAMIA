#!/bin/bash
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python
LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"

echo "$(date '+%F %T') test-set evaluation (5-fold ensemble on 306 held-out cases) started, PID $$" >> "$STATUS"
CUDA_VISIBLE_DEVICES=1 taskset -c 0-39 "$PYTHON" run_test_evaluation.py > "$LOGDIR/test_evaluation.log" 2>&1
rc=$?
echo "$(date '+%F %T') test-set evaluation finished (exit code ${rc})" >> "$STATUS"
