#!/bin/bash
#SBATCH --job-name=cow-r3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/08f_r3_%a_%A.out
#SBATCH --error=logs/08f_r3_%a_%A.out
# E6: rung-3 students (r3 s100-102) on super teacher v2. The thesis reports this rung
# regresses slightly; reproducing the regression IS the convergence evidence.
# Submit with:  bash hpc/submit.sh hpc/08f_rung3.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1
test -e "_super_teacher2_emb.npz" || { echo "missing _super_teacher2_emb.npz (run 08e)" >&2; exit 1; }
SEED=$((100 + SLURM_ARRAY_TASK_ID))
CKPT="_vitb_r3_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_mega.py --seed "${SEED}" --n-stage 2 --P 6 --teacher-npz _super_teacher2_emb.npz
