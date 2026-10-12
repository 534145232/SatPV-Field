#!/usr/bin/env bash
set -euo pipefail

# Create a PEP-668-safe project environment while retaining the CUDA-enabled
# PyTorch shipped by pytorch:2.11.0-cuda13.0-cudnn9-devel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-${SCRIPT_DIR}}"
VENV_DIR="${2:-${PROJECT_DIR}/.venv}"
BASE_PYTHON="${SATPV_BASE_PYTHON:-${BIGPV_BASE_PYTHON:-python}}"

if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: Python command not found: ${BASE_PYTHON}" >&2
  echo "Set SATPV_BASE_PYTHON to the interpreter supplied by the PyTorch image." >&2
  exit 1
fi

if ! "${BASE_PYTHON}" -c "import torch; assert torch.version.cuda is not None"; then
  echo "ERROR: ${BASE_PYTHON} cannot import the image's CUDA-enabled PyTorch." >&2
  echo "Run 'which -a python python3' and set SATPV_BASE_PYTHON to the correct interpreter." >&2
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/requirements.txt" ]]; then
  echo "ERROR: requirements.txt not found under ${PROJECT_DIR}" >&2
  exit 1
fi

"${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
"${VENV_PYTHON}" -m pip install --no-cache-dir \
  -r "${PROJECT_DIR}/requirements.txt"
"${VENV_PYTHON}" -m pip check

"${VENV_PYTHON}" -c "import torch; print('torch=', torch.__version__); print('torch CUDA=', torch.version.cuda); print('CUDA available=', torch.cuda.is_available()); print('GPU=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo
echo "Environment ready. Activate it with:"
echo "source ${VENV_DIR}/bin/activate"
