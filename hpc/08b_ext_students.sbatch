#!/bin/bash
#SBATCH --job-name=cow-ext
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/08b_ext_%a_%A.out
#SBATCH --error=logs/08b_ext_%a_%A.out
# Extension rung E2: seven Mega variants.
#   task 0   : megaft s50    (fused teacher, n-stage 1)
#   tasks 1-3: mega2 s60-62  (v4 teacher,    n-stage 2)
#   tasks 4-6: mega2ft s80-82 (fused teacher, n-stage 2)
# The teacher spaces for the historical runs were not archived; this mapping is the naming
# grid ({n-stage 1,2} x {v4, fused}). Verify each trio's sweep against sweep_*_v1.json.
# Submit with:  bash hpc/submit.sh hpc/08b_ext_students.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1
test -e "_imgcache.npy" || { echo "missing _imgcache.npy (run 01)" >&2; exit 1; }
test -e "_fused_teacher_emb.npz" || { echo "missing _fused_teacher_emb.npz (run 08a)" >&2; exit 1; }
test -e "_vitb_dst_emb_v4.npz" || { echo "missing _vitb_dst_emb_v4.npz (run 05b)" >&2; exit 1; }

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        SEED=50; NS=1; TN="_fused_teacher_emb.npz"; PP=10
        CKPT="_vitb_megaft_s50_ckpt.pt" ;;
    1|2|3)
        SEED=$((59 + SLURM_ARRAY_TASK_ID)); NS=2; TN="_vitb_dst_emb_v4.npz"; PP=6
        CKPT="_vitb_mega2_s${SEED}_ckpt.pt" ;;
    4|5|6)
        SEED=$((76 + SLURM_ARRAY_TASK_ID)); NS=2; TN="_fused_teacher_emb.npz"; PP=6
        CKPT="_vitb_mega2ft_s${SEED}_ckpt.pt" ;;
esac
# P=6 for n-stage 2 is the documented recipe (thesis ch5/appendix): two unfrozen Swin
# stages double the optimiser state, and P=10 out-of-memories even a 24 GB card.
JOB_SECONDS=$((12 * 3600))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== seed ${SEED}  n-stage ${NS}  teacher ${TN} -> ${CKPT}"
train_until 1000 "${CKPT}" 1800 "${JOB_SECONDS}" \
    python repro/vitb_unsup_mega.py --seed "${SEED}" --n-stage "${NS}" --teacher-npz "${TN}" --P "${PP}"
