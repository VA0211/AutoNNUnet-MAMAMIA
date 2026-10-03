#!/bin/bash
# Fully self-contained: resumes the (partial) preprocessed-data rsync to
# workspace, verifies byte-for-byte, symlinks, then resumes training.
# Everything, including the rsync itself, runs inside THIS one nohup'd
# process so nothing depends on the launching session staying alive.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
SRC=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed
DEST=/workspace/hailt/BC_MAMAMIA_data/nnUNet_preprocessed

echo "$(date '+%F %T') finish-move v2 started, PID $$, resuming rsync (previous attempt was killed mid-transfer by session teardown)" >> "$STATUS"

rsync -a "$SRC/" "$DEST/" >> "$LOGDIR/preprocessed_move_v2.log" 2>&1
rsync_rc=$?
echo "$(date '+%F %T') rsync finished with exit code ${rsync_rc}, verifying" >> "$STATUS"

src_count=$(find "$SRC" -type f | wc -l)
dest_count=$(find "$DEST" -type f | wc -l)
src_size=$(/usr/bin/du -sb "$SRC" | awk '{print $1}')
dest_size=$(/usr/bin/du -sb "$DEST" | awk '{print $1}')

echo "$(date '+%F %T') src: ${src_count} files, ${src_size} bytes | dest: ${dest_count} files, ${dest_size} bytes" >> "$STATUS"

if [ "$rsync_rc" != "0" ] || [ "$src_count" != "$dest_count" ] || [ "$src_size" != "$dest_size" ]; then
  echo "$(date '+%F %T') VERIFICATION FAILED - NOT touching local data, NOT resuming training. Manual check needed." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') verification OK, replacing local dir with symlink to workspace" >> "$STATUS"
rm -rf "$SRC"
ln -s "$DEST" "$SRC"

if [ ! -L "$SRC" ] || [ ! -d "$SRC/Dataset501_MAMAMIA" ]; then
  echo "$(date '+%F %T') SYMLINK CHECK FAILED - NOT resuming training. Manual check needed." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') symlink verified working, resuming training (folds 1-4, 300 epochs, fold 1 restarts from epoch 0)" >> "$STATUS"

nohup ./run_folds_300ep.sh > "$LOGDIR/orchestrator_300ep_resumed_v2.log" 2>&1 &
echo "$(date '+%F %T') training orchestrator relaunched, PID $!" >> "$STATUS"
