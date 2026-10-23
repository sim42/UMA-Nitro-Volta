#!/usr/bin/env bash
#SBATCH --job-name=uma-volta
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=00:45:00
#SBATCH --output=uma-volta-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch from the repository root}"
export PYTHONPATH="$PWD/packages/fairchem-core/src${PYTHONPATH:+:$PYTHONPATH}"
export UMA_NITRO_ENABLE_VOLTA=1
export OMP_NUM_THREADS=4
# A conda-backed venv may need its base C++ runtime (not a system library change).
base_prefix=$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')
export LD_LIBRARY_PATH="$base_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python experiments/volta/probe.py
.venv/bin/python experiments/volta/validate.py "${UMA_CHECKPOINT:-checkpoints/uma-s-1p2p1.pt}" "$@"
