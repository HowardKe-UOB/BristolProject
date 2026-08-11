#!/bin/bash
#SBATCH --job-name=cow-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/01_prep_%j.out
#SBATCH --error=logs/01_prep_%j.out
# Step 1: free signals, frozen splits, and the 15 GB uint8 image cache.
# CPU only: this step resizes crops, it trains nothing.
# Submit with:  bash hpc/submit.sh hpc/01_signals_cache.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh

echo "== phase 1: manifest, tracklets, mined pairs, splits"
python repro/build_phase1.py \
    --listing 2025Sep18.listing.txt \
    --tar     2025Sep18.tar.gz \
    --out     artifacts

echo "== image cache: crops resized to 518x518, mmap'd by every trainer (15 GB)"
python lib/vitb_unsup.py --build-cache \
    --listing 2025Sep18.listing.txt \
    --tar     2025Sep18.tar.gz \
    --work    _crops_train

ls -lh _imgcache.npy
echo "== done. next:  bash hpc/submit.sh hpc/02_vits_cache.sbatch"
