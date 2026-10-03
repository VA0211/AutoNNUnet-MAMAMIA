#!/bin/bash
# Reverts nnUNet_preprocessed back to local disk after NFS instability.
# Every NFS-touching step is timeout-guarded so a fresh hang fails fast
# and logs clearly, instead of blocking forever like the last attempt.

set -u
LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
LINK=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed
WORKSPACE_SRC=/workspace/hailt/BC_MAMAMIA_data/nnUNet_preprocessed
LOCAL_DEST=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed_local

echo "$(date '+%F %T') revert-to-local v2 started, PID $$ (timeout-guarded this time)" >> "$STATUS"

cd /home/hailt/BC_MAMAMIA/AutoNNUnet

if ! timeout 20 ls /workspace/hailt/miniconda3/etc/profile.d/conda.sh > /dev/null 2>&1; then
  echo "$(date '+%F %T') PREFLIGHT FAILED - NFS not responding for conda.sh within 20s. Aborting, NOT resuming training." >> "$STATUS"
  exit 1
fi

source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

if ! timeout 20 ls "$WORKSPACE_SRC" > /dev/null 2>&1; then
  echo "$(date '+%F %T') PREFLIGHT FAILED - NFS not responding for ${WORKSPACE_SRC} within 20s. Aborting, NOT resuming training." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') preflight checks passed, starting copy-back (timeout 60min)" >> "$STATUS"

timeout 3600 rsync -a "$WORKSPACE_SRC/" "$LOCAL_DEST/" >> "$LOGDIR/revert_copy_v2.log" 2>&1
rsync_rc=$?
echo "$(date '+%F %T') copy-back rsync finished with exit code ${rsync_rc} (124 = timed out) verifying" >> "$STATUS"

if [ "$rsync_rc" = "124" ]; then
  echo "$(date '+%F %T') RSYNC TIMED OUT - likely hung again on NFS. NOT resuming training. Manual check needed." >> "$STATUS"
  exit 1
fi

src_count=$(timeout 30 find "$WORKSPACE_SRC" -type f 2>/dev/null | wc -l)
dest_count=$(find "$LOCAL_DEST" -type f | wc -l)
src_size=$(timeout 30 /usr/bin/du -sb "$WORKSPACE_SRC" 2>/dev/null | awk '{print $1}')
dest_size=$(/usr/bin/du -sb "$LOCAL_DEST" | awk '{print $1}')

echo "$(date '+%F %T') workspace: ${src_count} files, ${src_size} bytes | local: ${dest_count} files, ${dest_size} bytes" >> "$STATUS"

if [ "$rsync_rc" != "0" ] || [ -z "$src_count" ] || [ "$src_count" != "$dest_count" ] || [ "$src_size" != "$dest_size" ]; then
  echo "$(date '+%F %T') VERIFICATION FAILED - leaving symlink pointed at workspace, NOT resuming training. Manual check needed." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') verification OK, swapping symlink for local real directory" >> "$STATUS"
rm -f "$LINK"
mv "$LOCAL_DEST" "$LINK"

if [ -L "$LINK" ] || [ ! -d "$LINK/Dataset501_MAMAMIA" ]; then
  echo "$(date '+%F %T') LOCAL SWAP CHECK FAILED - NOT resuming training. Manual check needed." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') local data confirmed in place, resuming training (folds 1-4, 300 epochs, restarting from fold 1 epoch 0)" >> "$STATUS"

nohup ./run_folds_300ep.sh > "$LOGDIR/orchestrator_300ep_local.log" 2>&1 &
echo "$(date '+%F %T') training orchestrator relaunched, PID $!" >> "$STATUS"
