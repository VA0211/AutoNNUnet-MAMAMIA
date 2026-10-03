#!/bin/bash
set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
PYTHON=/home/hailt/BC_MAMAMIA/AutoNNUnet/.venv_local/bin/python
LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"

echo "$(date '+%F %T') full postproc-variant rerun (with per-case CSV) started, PID $$" >> "$STATUS"

taskset -c 0-39 "$PYTHON" run_postproc_ablation.py \
  output/predictions/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres \
  "ConvolutionalEncoder (5-fold ensemble)" \
  > "$LOGDIR/rerun_postproc_convencoder.log" 2>&1

taskset -c 0-39 "$PYTHON" run_postproc_ablation.py \
  output/predictions/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres \
  "ResidualEncoderL (5-fold ensemble)" \
  > "$LOGDIR/rerun_postproc_resencl.log" 2>&1

taskset -c 0-39 "$PYTHON" run_laterality_postproc.py \
  output/predictions/baseline_ConvolutionalEncoder/Dataset501_MAMAMIA/3d_fullres \
  "ConvolutionalEncoder (5-fold ensemble)" \
  > "$LOGDIR/rerun_laterality_convencoder.log" 2>&1

taskset -c 0-39 "$PYTHON" run_laterality_postproc.py \
  output/predictions/baseline_ResidualEncoderL/Dataset501_MAMAMIA/3d_fullres \
  "ResidualEncoderL (5-fold ensemble)" \
  > "$LOGDIR/rerun_laterality_resencl.log" 2>&1

echo "$(date '+%F %T') full postproc-variant rerun finished" >> "$STATUS"
