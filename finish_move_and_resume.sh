#!/bin/bash
# Waits for the in-flight preprocessed-data rsync to workspace to finish,
# verifies it byte-for-byte, replaces the local copy with a symlink, then
# relaunches the folds 1-4 (300-epoch) training orchestrator. Fully
# self-contained and detached so it completes even if the session/laptop
# connection that started it goes away.

set -u
cd /home/hailt/BC_MAMAMIA/AutoNNUnet
source /workspace/hailt/miniconda3/etc/profile.d/conda.sh
conda activate autonnunet

LOGDIR=/home/hailt/BC_MAMAMIA/AutoNNUnet/fold_run_logs
STATUS="$LOGDIR/status.log"
SRC=/home/hailt/BC_MAMAMIA/AutoNNUnet/data/nnUNet_preprocessed
DEST=/workspace/hailt/BC_MAMAMIA_data/nnUNet_preprocessed

echo "$(date '+%F %T') finish-move watcher started, PID $$, waiting for preprocessed-data rsync to finish" >> "$STATUS"

# Wait until no rsync process targeting this destination is still running
while pgrep -f "rsync.*nnUNet_preprocessed" > /dev/null; do
  sleep 15
done

echo "$(date '+%F %T') rsync process finished, verifying" >> "$STATUS"

src_count=$(find "$SRC" -type f | wc -l)
dest_count=$(find "$DEST" -type f | wc -l)
src_size=$(/usr/bin/du -sb "$SRC" | awk '{print $1}')
dest_size=$(/usr/bin/du -sb "$DEST" | awk '{print $1}')

echo "$(date '+%F %T') src: ${src_count} files, ${src_size} bytes | dest: ${dest_count} files, ${dest_size} bytes" >> "$STATUS"

if [ "$src_count" != "$dest_count" ] || [ "$src_size" != "$dest_size" ]; then
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

nohup ./run_folds_300ep.sh > "$LOGDIR/orchestrator_300ep_resumed.log" 2>&1 &
echo "$(date '+%F %T') training orchestrator relaunched, PID $!" >> "$STATUS"
