#!/bin/bash
#SBATCH --job-name=nowcast_dataprep
#SBATCH --output=logs/dataprep_%j.log
#SBATCH --error=logs/dataprep_%j.log
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "Running on node: $(hostname)"

cd /home/YOUR_USERNAME/thunderstorm_alert/hpc_train

# No GPU needed for downloading/preprocessing — this is pure I/O + CPU.
# If your cluster needs a venv activated per-job (rather than a persistent
# one), uncomment and adjust:
# module load python-3.11
# source ../.venv/bin/activate

python data_prep/download_sevir.py --min-events 2000 --out-dir data/raw/sevir
python data_prep/preprocess.py --sevir-dir data/raw/sevir --out-dir data/processed/sevir --min-events 2000

echo "Done. Processed event count:"
ls data/processed/sevir | wc -l
