#!/bin/bash
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python
LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"

echo "$(date '+%F %T') official challenge score (DSC+NormHD+PerformanceScore) computation started, PID $$" >> "$STATUS"

taskset -c 0-39 "$PYTHON" run_official_challenge_score.py \
  output/predictions/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres \
  "ConvolutionalEncoder (5-fold ensemble)" \
  > "$LOGDIR/official_score_convencoder.log" 2>&1

taskset -c 0-39 "$PYTHON" run_official_challenge_score.py \
  output/predictions/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres \
  "ResidualEncoderL (5-fold ensemble)" \
  > "$LOGDIR/official_score_resencl.log" 2>&1

taskset -c 0-39 "$PYTHON" run_official_challenge_score.py \
  output/predictions/baseline_ResidualEncoderL_fold1only/Dataset501_MAMAMIA/3d_fullres \
  "ResidualEncoderL (fold 1 only)" \
  > "$LOGDIR/official_score_resencl_fold1only.log" 2>&1

echo "$(date '+%F %T') official challenge score computation finished" >> "$STATUS"
