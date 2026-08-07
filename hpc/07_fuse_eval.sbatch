#!/bin/bash
#SBATCH --job-name=cow-fuse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/07_fuse_%j.out
#SBATCH --error=logs/07_fuse_%j.out
# Fusion and evaluation. CPU only: every model has already written its embeddings, so this
# step is arithmetic over saved vectors. Produces the headline numbers on all three
# protocols and rewrites the JSONs under artifacts2/.
# Submit with:  bash hpc/submit.sh hpc/07_fuse_eval.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh

for f in _sweep_final_zerohuman_emb.npz _sweep_mega_trio_emb.npz _vitb_dst_emb_v4.npz; do
    test -e "$f" || { echo "missing prerequisite: $f (run hpc/06b_embed_sweep.sbatch first)" >&2; exit 1; }
done

echo "== keeping the committed reference results so your run can be compared against them"
stamp="$(date +%Y%m%d_%H%M)"
mkdir -p "artifacts2_reference_${stamp}"
cp artifacts2/*.json "artifacts2_reference_${stamp}/"

echo "== heterogeneous fusion (DINOv2 students + MegaDescriptor students)"
python repro/fuse_hetero.py

echo "== final operating point, greedy selection over the student pool"
python repro/fuse_final.py

echo "== protocol audit"
python repro/validate_protocols.py

cat <<EOF

== done. Compare your numbers with the archived ones:

    diff <(python -m json.tool artifacts2_reference_${stamp}/fuse_final_v1.json) \\
         <(python -m json.tool artifacts2/fuse_final_v1.json)

Agreement is expected within the reported seed variance (students +-0.03 rank-1), not to
the last decimal: a different GPU and data order move the fifth digit.
EOF
