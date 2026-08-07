#!/bin/bash
#SBATCH --job-name=cow-select
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/08g_select_%j.out
#SBATCH --error=logs/08g_select_%j.out
# E7+E8: embed the rung-3 trio, then run every selection over the full zoo.
# The headline lives in mega_search.py's max_P1 entry: archived value 0.9448, which the
# dissertation rounds to 0.945.
# Submit with:  bash hpc/submit.sh hpc/08g_selection.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1
for s in 100 101 102; do
    test -f "_vitb_r3_s${s}_ckpt.pt" || { echo "missing r3 s${s} (run 08f)" >&2; exit 1; }
done
python repro/eval_sweep.py --tag r3_trio --backbone mega \
    --ckpts _vitb_r3_s100_ckpt.pt _vitb_r3_s101_ckpt.pt _vitb_r3_s102_ckpt.pt

echo "== snapshotting reference JSONs before the selection scripts rewrite them"
stamp="$(date +%Y%m%d_%H%M)"
mkdir -p "artifacts2_reference_${stamp}"
cp artifacts2/*.json "artifacts2_reference_${stamp}/"

python repro/mega_search.py
python experiments/ablations/mega_search2.py
python repro/ensemble_search.py
python repro/fuse_final.py

echo "== headline check: the paper's max_P1 reference is 0.9448 =="
python -c "import json; d=json.load(open('artifacts2/mega_search_v1.json')); print('cluster max_P1:', d.get('max_P1'))"
