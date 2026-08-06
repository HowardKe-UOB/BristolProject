#!/bin/bash
#SBATCH --job-name=cow-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/00_smoke_%j.out
#SBATCH --error=logs/00_smoke_%j.out
# Two minutes that can save a day. Checks, on a real compute node, that:
#   1. the installed torch can talk to this node's driver (a CUDA build newer than the
#      driver installs fine and only fails here, at the first real GPU call);
#   2. one training step of the actual recipe fits in this card's memory.
# Run this before submitting anything else.
#   bash hpc/submit.sh hpc/00_smoke_gpu.sbatch
set -euo pipefail
mkdir -p logs
source hpc/env.sh
export HF_HUB_OFFLINE=1

echo "== node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import sys
import torch

print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\nFAIL: torch cannot see a GPU on this node.")
    print("Most often the installed CUDA build is newer than the node's driver.")
    print("Fix: reinstall a matching build inside .venv, e.g. for a CUDA 12.1 driver")
    print("  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  device: {name}, {total:.1f} GB")

try:
    a = torch.randn(2048, 2048, device=dev)
    torch.cuda.synchronize()
    print(f"  arithmetic ok (matmul mean {(a @ a).mean().item():.4f})")
except RuntimeError as exc:
    print(f"\nFAIL at the first real GPU call: {exc}")
    sys.exit(1)

# One step of the actual recipe: P*K tracklets x T frames = 12*4*2 = 96 crops at 518x518
# through ViT-B/14, fp16 autocast, gradients on the last four blocks.
import timm

print("\n== one training step of the real recipe (96 crops of 518x518 through ViT-B)")
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                          num_classes=0, img_size=518).to(dev)
for p in model.parameters():
    p.requires_grad_(False)
for blk in model.blocks[-4:]:
    for p in blk.parameters():
        p.requires_grad_(True)

x = torch.randn(96, 3, 518, 518, device=dev)
torch.cuda.reset_peak_memory_stats()
try:
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x)
        loss = out.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
except torch.cuda.OutOfMemoryError:
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nOUT OF MEMORY on a {total:.1f} GB card (reached {peak:.1f} GB).")
    print("Either ask for a bigger card:")
    print("    sed -i 's/gpu:[^\"]*1/gpu:a100:1/' hpc/env.sh")
    print("or halve the batch by adding --P 6 to the trainers, and report the change,")
    print("because a different batch size is a different training recipe.")
    sys.exit(1)

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  step ok. peak memory {peak:.1f} GB of {total:.1f} GB "
      f"({100 * peak / total:.0f}% used, {total - peak:.1f} GB spare)")
print("\nPASS: this node can run the ladder as configured.")
PY

echo
echo "== next:  bash hpc/submit.sh hpc/01_signals_cache.sbatch"
