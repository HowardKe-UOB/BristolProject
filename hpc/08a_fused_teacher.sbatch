#!/bin/bash
#SBATCH --job-name=cow-fteach
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/08a_fteach_%j.out
#SBATCH --error=logs/08a_fteach_%j.out
# Extension rung E1: the fused teacher (0.4*DINOv2 + 0.6*Mega trio-mean concatenation).
# Pure numpy over saved embeddings: CPU. Writes _fused_teacher_emb.npz (key t0).
# Submit with:  bash hpc/submit.sh hpc/08a_fused_teacher.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
test -e "_vitb_dst_emb_v4.npz" || { echo "missing _vitb_dst_emb_v4.npz (run 05b)" >&2; exit 1; }
test -e "_sweep_mega_trio_emb.npz" || { echo "missing _sweep_mega_trio_emb.npz (run 06b)" >&2; exit 1; }
python repro/make_fused_teacher.py
ls -lh _fused_teacher_emb.npz
