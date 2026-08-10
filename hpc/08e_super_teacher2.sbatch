#!/bin/bash
#SBATCH --job-name=cow-steach2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/08e_steach2_%j.out
#SBATCH --error=logs/08e_steach2_%j.out
# E5: embed the sup2 trio, then build the rung-3 super teacher
# (sup2 + mega2ft + hc -> _super_teacher2_emb.npz).
# Submit with:  bash hpc/submit.sh hpc/08e_super_teacher2.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1
require_trained 1000 _vitb_sup2_s90_ckpt.pt _vitb_sup2_s91_ckpt.pt _vitb_sup2_s92_ckpt.pt
python repro/eval_sweep.py --tag sup2_trio --backbone mega \
    --ckpts _vitb_sup2_s90_ckpt.pt _vitb_sup2_s91_ckpt.pt _vitb_sup2_s92_ckpt.pt
python repro/make_super_teacher2.py
ls -lh _super_teacher2_emb.npz
