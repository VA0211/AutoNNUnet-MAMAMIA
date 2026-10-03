#!/bin/bash
# Reverts nnUNet_preprocessed back to local disk after NFS instability
# caused training to hang twice. Copies from workspace back to a local
# path, verifies, replaces the symlink with the real local directory,
# then resumes training. Fully detached (setsid), no dependency on the
# launching session.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
LINK=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed
WORKSPACE_SRC=/workspace/hailt/BC_MAMAMIA_data/nnUNet_preprocessed
LOCAL_DEST=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed_local

echo "$(date '+%F %T') revert-to-local started, PID $$ (NFS caused 2 hangs, moving preprocessed data back to local disk)" >> "$STATUS"

rsync -a "$WORKSPACE_SRC/" "$LOCAL_DEST/" >> "$LOGDIR/revert_copy.log" 2>&1
rsync_rc=$?
echo "$(date '+%F %T') copy-back rsync finished with exit code ${rsync_rc}, verifying" >> "$STATUS"

src_count=$(find "$WORKSPACE_SRC" -type f | wc -l)
dest_count=$(find "$LOCAL_DEST" -type f | wc -l)
src_size=$(/usr/bin/du -sb "$WORKSPACE_SRC" | awk '{print $1}')
dest_size=$(/usr/bin/du -sb "$LOCAL_DEST" | awk '{print $1}')

echo "$(date '+%F %T') workspace: ${src_count} files, ${src_size} bytes | local: ${dest_count} files, ${dest_size} bytes" >> "$STATUS"

if [ "$rsync_rc" != "0" ] || [ "$src_count" != "$dest_count" ] || [ "$src_size" != "$dest_size" ]; then
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
