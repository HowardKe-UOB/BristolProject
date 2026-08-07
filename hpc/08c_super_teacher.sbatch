#!/bin/bash
#SBATCH --job-name=cow-steach1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/08c_steach1_%j.out
#SBATCH --error=logs/08c_steach1_%j.out
# E3: embed the extension students, then build the rung-2 super teacher
# (mega2ft 0.45 + mega2 0.35 + hc trio 0.20 -> _super_teacher_emb.npz, key t0).
# Submit with:  bash hpc/submit.sh hpc/08c_super_teacher.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1
for f in _vitb_megaft_s50_ckpt.pt _vitb_mega2_s60_ckpt.pt _vitb_mega2_s61_ckpt.pt \
         _vitb_mega2_s62_ckpt.pt _vitb_mega2ft_s80_ckpt.pt _vitb_mega2ft_s81_ckpt.pt \
         _vitb_mega2ft_s82_ckpt.pt; do
    test -f "$f" || { echo "missing $f (run 08b)" >&2; exit 1; }
done
python repro/eval_sweep.py --tag megaft_s50 --backbone mega --ckpts _vitb_megaft_s50_ckpt.pt
python repro/eval_sweep.py --tag mega2_trio --backbone mega \
    --ckpts _vitb_mega2_s60_ckpt.pt _vitb_mega2_s61_ckpt.pt _vitb_mega2_s62_ckpt.pt
python repro/eval_sweep.py --tag mega2ft_trio --backbone mega \
    --ckpts _vitb_mega2ft_s80_ckpt.pt _vitb_mega2ft_s81_ckpt.pt _vitb_mega2ft_s82_ckpt.pt
python repro/make_super_teacher.py
ls -lh _super_teacher_emb.npz
