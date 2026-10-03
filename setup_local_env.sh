#!/bin/bash
# Builds a fully local (non-NFS) Python environment for AutoNNUnet.
# Verified working end-to-end (torch+cuda, AutoNNUNetTrainer import).

set -u
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO=/home/hailt/BC_MAMAMIA/AutoNNUnet
VENV="$REPO/.venv_local"
LOGDIR="$REPO/fold_run_logs"
STATUS="$LOGDIR/status.log"
LOG="$LOGDIR/setup_local_env.log"

echo "$(date '+%F %T') local env setup started, PID $$" >> "$STATUS"

(
  set -e
  echo "=== creating venv ==="
  /usr/bin/python3.12 -m venv "$VENV"

  PIP="$VENV/bin/pip"
  PY="$VENV/bin/python"

  echo "=== upgrading pip ==="
  "$PIP" install --upgrade pip

  echo "=== installing setuptools + wheel ==="
  "$PIP" install setuptools wheel

  echo "=== pre-installing pinned numpy ==="
  "$PIP" install numpy==1.26.4

  echo "=== installing local submodules ==="
  cd "$REPO/submodules/batchgenerators" && "$PIP" install .
  cd "$REPO/submodules/hypersweeper" && "$PIP" install -e . --no-deps
  cd "$REPO/submodules/neps" && "$PIP" install . --ignore-requires-python --no-deps
  cd "$REPO/submodules/nnUNet" && "$PIP" install .

  echo "=== re-pinning numpy to 1.26.4 ==="
  "$PIP" install numpy==1.26.4

  echo "=== installing AutoNNUnet + core dependencies from pyproject.toml ==="
  cd "$REPO" && "$PIP" install -e .

  echo "=== final numpy re-pin ==="
  "$PIP" install numpy==1.26.4

  echo "=== restoring ConfigSpace pin and adding neps's missing nltk dep (--no-deps skipped it) ==="
  "$PIP" install ConfigSpace==1.2.0 nltk

  echo "=== verifying the environment ==="
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
  "$PY" -c "import numpy; print('numpy', numpy.__version__)"
  "$PY" -c "from autonnunet.training.auto_nnunet_trainer import AutoNNUNetTrainer; print('AutoNNUNetTrainer import OK')"

) > "$LOG" 2>&1
SETUP_RC=$?

if [ "$SETUP_RC" != "0" ]; then
  echo "$(date '+%F %T') LOCAL ENV SETUP FAILED (exit ${SETUP_RC}) - see setup_local_env.log. NOT touching training." >> "$STATUS"
  exit 1
fi

echo "$(date '+%F %T') local env setup completed successfully, venv at $VENV" >> "$STATUS"
