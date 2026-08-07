#!/bin/bash
#SBATCH --job-name=cow-mega
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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

# Fail fast and legibly if a previous stage has not produced its output: a queued job
# that starts before its input exists otherwise dies in a Python traceback hours later.
test -e "artifacts2/distill_labels_v1.json" || { echo "missing prerequisite: artifacts2/distill_labels_v1.json (run hpc/04_teacher.sbatch first)" >&2; exit 1; }
test -e "_imgcache.npy" || { echo "missing prerequisite: _imgcache.npy (run hpc/01_signals_cache.sbatch first)" >&2; exit 1; }
test -e "_vitb_dst_emb_v4.npz" || { echo "missing prerequisite: _vitb_dst_emb_v4.npz (run hpc/05a then 05b first)" >&2; exit 1; }

SEED=$((40 + SLURM_ARRAY_TASK_ID))
CKPT="_vitb_mega_s${SEED}_ckpt.pt"
JOB_SECONDS=$((12 * 3600))

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_mega.py --seed "${SEED}"

echo "== when all three are done:  bash hpc/submit.sh hpc/07_fuse_eval.sbatch"
