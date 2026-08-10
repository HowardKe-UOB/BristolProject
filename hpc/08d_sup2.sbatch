#!/bin/bash
#SBATCH --job-name=cow-sup2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/08d_sup2_%a_%A.out
#SBATCH --error=logs/08d_sup2_%a_%A.out
# E4: rung-2 students (sup2 s90-92) trained in the super-teacher space. n-stage 2 is an
# assumption of the reconstruction; verify the trio's sweep against sweep_sup2_trio_v1.json.
# Submit with:  bash hpc/submit.sh hpc/08d_sup2.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1
test -e "_imgcache.npy" || { echo "missing _imgcache.npy (run 01)" >&2; exit 1; }
test -e "_super_teacher_emb.npz" || { echo "missing _super_teacher_emb.npz (run 08c)" >&2; exit 1; }
SEED=$((90 + SLURM_ARRAY_TASK_ID))
CKPT="_vitb_sup2_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_mega.py --seed "${SEED}" --n-stage 2 --P 6 --teacher-npz _super_teacher_emb.npz
