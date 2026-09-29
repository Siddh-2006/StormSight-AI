#!/bin/bash
#SBATCH --job-name=nowcast_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

echo "Running on node: $(hostname)"
module load cuda-12.8

cd /home/YOUR_USERNAME/thunderstorm_alert/hpc_train/model

python train.py \
  --data-source mixed \
  --epochs 60 \
  --episodes-per-epoch 500 \
  --batch-size 32 \
  --hidden-channels 32,64 \
  --num-workers 8
