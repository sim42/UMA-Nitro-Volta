#!/usr/bin/env bash
# Run from the repository root, outside a GPU allocation.
set -euo pipefail
python_bin=${PYTHON:-python3.11}
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r experiments/volta/requirements.txt
# Use PYTHONPATH in run-slurm.sh: upstream packaging pins a different Torch version.
