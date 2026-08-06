#!/bin/bash
# One-time setup, run on a BluePebble LOGIN node (compute nodes may have no internet).
#   bash hpc/setup_env.sh
set -euo pipefail

source "$(dirname "$0")/env.sh"

echo "== python: $(python --version) at $(command -v python)"

if [ ! -d .venv ]; then
    python -m venv .venv
    echo "== created .venv"
fi
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo "== dependencies installed"

mkdir -p "${HF_HOME}" "${TORCH_HOME}"

echo "== pre-downloading backbones into ${HF_HOME} (this is why setup runs on the login node)"
python - <<'PY'
import timm, torch

# DINOv2 ViT-B/14: the main backbone, Stages 1-3.
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
print("  DINOv2 ViT-B/14 ok, params", sum(p.numel() for p in m.parameters()) / 1e6, "M")

# DINOv2 ViT-S/14: the frozen features Stage 1 seeds its proxies from.
s = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
print("  DINOv2 ViT-S/14 ok")

# MegaDescriptor: Stage 4. Skipped with a warning if the hub name has moved.
try:
    g = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
    print("  MegaDescriptor-L-384 ok")
except Exception as exc:
    print("  MegaDescriptor download FAILED:", exc)
    print("  Stage 4 will not run offline until this is cached.")

print("  torch", torch.__version__, "cuda build", torch.version.cuda)
PY

cat <<EOF

== setup complete.

Check the dataset is in place, then submit the first job:

    ls -lh 2025Sep18.tar.gz 2025Sep18.listing.txt
    sbatch hpc/01_signals_cache.sbatch
EOF
