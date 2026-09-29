#!/bin/bash
# SLURM batch script to train the ConvLSTM nowcaster on HPC with a much larger
# budget than the local 8GB-RAM dev machine can handle. See scripts/HPC_TRAINING.md
# for the full workflow (sync code, submit, retrieve checkpoint, re-evaluate locally).
#
# EDIT THE #SBATCH LINES BELOW for your specific cluster (partition/account names,
# walltime, GPU type/count are cluster-specific and cannot be guessed here).
# If your HPC uses PBS/Torque instead of SLURM, see the commented PBS variant
# at the bottom of scripts/HPC_TRAINING.md.

#SBATCH --job-name=nowcast-train
#SBATCH --partition=gpu                # EDIT: your GPU partition name
#SBATCH --account=YOUR_ACCOUNT         # EDIT: your allocation/account
#SBATCH --gres=gpu:1                   # EDIT: GPU count/type, e.g. gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00                # EDIT: walltime budget
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
mkdir -p logs

echo "== Environment setup =="
# EDIT: replace with your cluster's module system, e.g.:
#   module load python/3.11 cuda/12.4
# or activate a pre-existing conda/venv environment instead of the block below.

if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11
fi
uv pip install --python .venv/bin/python -r backend/requirements.txt

echo "== Training =="
cd backend
# Scale this up freely — HPC has the RAM/GPU the local dev machine doesn't.
# Suggested starting point for a real training run (adjust to your time budget):
../.venv/bin/python -u -m app.ml.train \
    --epochs 60 \
    --episodes-per-epoch 500 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-channels 32,64

echo "== Done. Checkpoint at models/checkpoints/convlstm.pt =="
echo "Copy it back to the dev machine and re-run scripts/evaluate.py there."
