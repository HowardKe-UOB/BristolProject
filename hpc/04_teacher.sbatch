#!/bin/bash
#SBATCH --job-name=cow-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/04_teacher_%j.out
#SBATCH --error=logs/04_teacher_%j.out
# Stage 2: average the five CAP seeds into a distance-ensemble teacher, then freeze the
# label set the students will be taught from (intra-camera clusters + pairwise cross-camera
# links). The labels are written once and never refreshed, which is what stops the feedback
# loop between a model and its own mistakes.
# Submit with:  bash hpc/submit.sh hpc/04_teacher.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
source hpc/train_loop.sh
export HF_HUB_OFFLINE=1

require_trained 1000 _vitb_cap_s0_ckpt.pt _vitb_cap_s1_ckpt.pt _vitb_cap_s2_ckpt.pt \
    _vitb_cap_s3_ckpt.pt _vitb_cap_s4_ckpt.pt

echo "== embedding the five seeds and fusing them (concat / RRF / distance-mean)"
python repro/eval_cap_ensemble.py

echo "== freezing the distillation labels"
python repro/make_distill_labels.py

echo "== done. next:  bash hpc/submit.sh hpc/05_students.sbatch"
