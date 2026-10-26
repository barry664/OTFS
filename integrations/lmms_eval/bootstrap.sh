#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash integrations/lmms_eval/bootstrap.sh /path/to/lmms-eval-otfs" >&2
  exit 2
fi

OTFS_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LMMS_ROOT=$1
COMMIT=ee31b5a65ea79c8a2a4c9d6fcab736751d39513d

git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git "$LMMS_ROOT"
git -C "$LMMS_ROOT" checkout "$COMMIT"
python "$OTFS_ROOT/scripts/apply_lmms_eval_integration.py" "$LMMS_ROOT"

cd "$LMMS_ROOT"
uv sync --frozen --python 3.11.14 --extra qwen --extra video

echo "LMMs-Eval is ready at $LMMS_ROOT"
echo "Activate it with: source $LMMS_ROOT/.venv/bin/activate"
