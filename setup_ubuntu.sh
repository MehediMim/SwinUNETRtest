#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_VARIANT="${TORCH_VARIANT:-cu118}"
case "$TORCH_VARIANT" in
  cu118|cpu) ;;
  *) echo "TORCH_VARIANT must be cu118 or cpu" >&2; exit 1 ;;
esac
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for these pinned dependencies"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.0.1 --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, monai; print("PyTorch:", torch.__version__, "MONAI:", monai.__version__, "CUDA available:", torch.cuda.is_available())'
echo 'Setup complete. Read UBUNTU.md before training.'
