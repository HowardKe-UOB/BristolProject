#!/bin/bash
#SBATCH --job-name=cow-students
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/05_student_%a_%A.out
#SBATCH --error=logs/05_student_%a_%A.out
# Stage 3: six students distilled from the frozen labels.
#   tasks 0-2: deployment students, seeds 10-12
#   tasks 3-5: hard cannot-link students, seeds 16-18  (the Stage-3 winner)
# Keep these seed numbers: fuse_final.py finds each model's embeddings by filename.
# Submit with:  bash hpc/submit.sh hpc/05_students.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1

# Fail fast and legibly if a previous stage has not produced its output: a queued job
# that starts before its input exists otherwise dies in a Python traceback hours later.
test -e "artifacts2/distill_labels_v1.json" || { echo "missing prerequisite: artifacts2/distill_labels_v1.json (run hpc/04_teacher.sbatch first)" >&2; exit 1; }
test -e "_imgcache.npy" || { echo "missing prerequisite: _imgcache.npy (run hpc/01_signals_cache.sbatch first)" >&2; exit 1; }

case "${SLURM_ARRAY_TASK_ID}" in
    0|1|2) SCRIPT="repro/vitb_unsup_deploy.py";  SEED=$((10 + SLURM_ARRAY_TASK_ID));     TAG="dep" ;;
    3|4|5) SCRIPT="repro/vitb_unsup_hardcl2.py"; SEED=$((16 + SLURM_ARRAY_TASK_ID - 3)); TAG="hc2" ;;
    *) echo "unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

CKPT="_vitb_${TAG}_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))

echo "== ${SCRIPT} seed ${SEED} -> ${CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python "${SCRIPT}" --seed "${SEED}"

echo "== when all six are done:  bash hpc/submit.sh hpc/06_mega.sbatch"
