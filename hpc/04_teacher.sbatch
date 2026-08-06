#!/bin/bash
#SBATCH --job-name=cow-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
export HF_HUB_OFFLINE=1

for s in 0 1 2 3 4; do
    test -f "_vitb_cap_s${s}_ckpt.pt" || { echo "missing _vitb_cap_s${s}_ckpt.pt" >&2; exit 1; }
done

echo "== embedding the five seeds and fusing them (concat / RRF / distance-mean)"
python repro/eval_cap_ensemble.py

echo "== freezing the distillation labels"
python repro/make_distill_labels.py

echo "== done. next:  bash hpc/submit.sh hpc/05_students.sbatch"
