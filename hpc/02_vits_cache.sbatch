#!/bin/bash
#SBATCH --job-name=cow-vits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/02_vits_%j.out
#SBATCH --error=logs/02_vits_%j.out
# Step 2: the frozen DINOv2 ViT-S features Stage 1 seeds its camera-aware proxies from.
# 1,079 tracklets x 8 frames x 384 dims, about 9 MB. Roughly 15 minutes on one GPU.
# Submit with:  bash hpc/submit.sh hpc/02_vits_cache.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1   # weights were fetched by setup_env.sh on the login node

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python repro/make_vits_cache.py \
    --listing 2025Sep18.listing.txt \
    --tar     2025Sep18.tar.gz

ls -lh dino_clip_feats_v1.npz
echo "== done. next:  bash hpc/submit.sh hpc/03_cap.sbatch"
