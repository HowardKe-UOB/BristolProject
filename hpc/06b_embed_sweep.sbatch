#!/bin/bash
#SBATCH --job-name=cow-embed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/06b_embed_%j.out
#SBATCH --error=logs/06b_embed_%j.out
# Between training and fusion: embed the trained students once, so that every fusion and
# evaluation step afterwards is CPU arithmetic over saved vectors. Produces
#   _sweep_final_zerohuman_emb.npz   (six DINOv2 students: dep 10-12 + hc2 16-18)
#   _sweep_mega_trio_emb.npz         (three Mega students: 40-42)
# which fuse_hetero.py and fuse_final.py read. Keep the checkpoint order below: the npz
# stores models as m0..mN in argument order, and fuse_final names them by position.
# Submit with:  bash hpc/submit.sh hpc/06b_embed_sweep.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1

for f in _vitb_dep_s10_ckpt.pt _vitb_dep_s11_ckpt.pt _vitb_dep_s12_ckpt.pt \
         _vitb_hc2_s16_ckpt.pt _vitb_hc2_s17_ckpt.pt _vitb_hc2_s18_ckpt.pt; do
    test -f "$f" || { echo "missing $f (run hpc/05_students.sbatch first)" >&2; exit 1; }
done
for f in _vitb_mega_s40_ckpt.pt _vitb_mega_s41_ckpt.pt _vitb_mega_s42_ckpt.pt; do
    test -f "$f" || { echo "missing $f (run hpc/06_mega.sbatch first)" >&2; exit 1; }
done

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== embedding the six DINOv2 students"
python repro/eval_sweep.py --tag final_zerohuman \
    --ckpts _vitb_dep_s10_ckpt.pt _vitb_dep_s11_ckpt.pt _vitb_dep_s12_ckpt.pt \
            _vitb_hc2_s16_ckpt.pt _vitb_hc2_s17_ckpt.pt _vitb_hc2_s18_ckpt.pt

echo "== embedding the three Mega students"
python repro/eval_sweep.py --tag mega_trio --backbone mega \
    --ckpts _vitb_mega_s40_ckpt.pt _vitb_mega_s41_ckpt.pt _vitb_mega_s42_ckpt.pt

ls -lh _sweep_final_zerohuman_emb.npz _sweep_mega_trio_emb.npz
echo "== done. next:  bash hpc/submit.sh hpc/07_fuse_eval.sbatch"
