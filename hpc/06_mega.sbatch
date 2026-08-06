#!/bin/bash
#SBATCH --job-name=cow-mega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/06_mega_%a_%A.out
#SBATCH --error=logs/06_mega_%a_%A.out
# Stage 4: the same frozen labels distilled into a different backbone family
# (MegaDescriptor, an animal-domain foundation model), seeds 40-42. DINOv2 students and
# Mega students fail on different animals, which is what makes fusing them worth doing.
# Submit with:  bash hpc/submit.sh hpc/06_mega.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1

SEED=$((40 + SLURM_ARRAY_TASK_ID))
CKPT="_vitb_mega_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_mega.py --seed "${SEED}"

echo "== when all three are done:  bash hpc/submit.sh hpc/07_fuse_eval.sbatch"
