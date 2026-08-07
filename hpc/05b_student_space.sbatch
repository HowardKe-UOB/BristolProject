#!/bin/bash
#SBATCH --job-name=cow-dstspace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/05b_space_%j.out
#SBATCH --error=logs/05b_space_%j.out
# Stage 3, between rungs: embed the five holdout students (flip TTA) into one npz,
# _vitb_dst_emb_v4.npz. The deployment students, the hard cannot-link students, the Mega
# students and validate_protocols.py all read this file; its k=2 trio keys (s7,s8,s9) are
# the mining space of the next rung.
# Submit with:  bash hpc/submit.sh hpc/05b_student_space.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1

for s in 5 6 7 8 9; do
    test -f "_vitb_dst_s${s}_ckpt.pt" || { echo "missing _vitb_dst_s${s}_ckpt.pt (run hpc/05a_holdout_students.sbatch first)" >&2; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python repro/fuse_student.py \
    --students _vitb_dst_s5_ckpt.pt _vitb_dst_s6_ckpt.pt _vitb_dst_s7_ckpt.pt \
               _vitb_dst_s8_ckpt.pt _vitb_dst_s9_ckpt.pt \
    --out-npz _vitb_dst_emb_v4.npz \
    --out-json artifacts2/fuse_student_v4.json

ls -lh _vitb_dst_emb_v4.npz
echo "== done. next:  bash hpc/submit.sh hpc/05_students.sbatch"
