#!/bin/bash
#SBATCH --job-name=cow-dst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/05a_dst_%a_%A.out
#SBATCH --error=logs/05a_dst_%a_%A.out
# Stage 3, first rung: the five HOLDOUT students (seeds 5-9), distilled from the frozen
# teacher labels. Seeds 5-6 use k=1 links, seeds 7-9 use k=2; the k=2 trio (s7,s8,s9)
# becomes the teacher space (_vitb_dst_emb_v4.npz) that the deployment students and the
# Mega students mine their links in. This rung was missing from the original job set,
# which is why 05_students failed with FileNotFoundError on that npz.
# Submit with:  bash hpc/submit.sh hpc/05a_holdout_students.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1

test -e "_vitb_cap_ens5_emb_v1.npz" || { echo "missing prerequisite: _vitb_cap_ens5_emb_v1.npz (run hpc/04_teacher.sbatch first)" >&2; exit 1; }
test -e "_imgcache.npy" || { echo "missing prerequisite: _imgcache.npy (run hpc/01_signals_cache.sbatch first)" >&2; exit 1; }

SEED=$((5 + SLURM_ARRAY_TASK_ID))
case "${SEED}" in
    5|6) LINKK=1 ;;
    7|8|9) LINKK=2 ;;
esac
CKPT="_vitb_dst_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== holdout student seed ${SEED}, link-k ${LINKK} -> ${CKPT}"

train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_distill.py --seed "${SEED}" --link-k "${LINKK}"

echo "== when all five are done:  bash hpc/submit.sh hpc/05b_student_space.sbatch"
