#!/bin/bash
#SBATCH --job-name=cow-cap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/03_cap_s%a_%A.out
#SBATCH --error=logs/03_cap_s%a_%A.out
# Stage 1: camera-aware proxy self-training, one array task per seed (0-4).
# These five models become the ensemble teacher in Stage 2.
#
# Needs a 24 GB GPU for the default batch (--P 12 --K 4 --T 2). On an 11 GB card add
# "--P 6" below and record the change: a different batch is a different recipe.
#
# Submit with:  bash hpc/submit.sh hpc/03_cap.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1

SEED="${SLURM_ARRAY_TASK_ID}"
CKPT="_vitb_cap_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_cap.py --seed "${SEED}"

echo "== seed ${SEED} finished. When all five are done:"
echo "==   bash hpc/submit.sh hpc/04_teacher.sbatch"
