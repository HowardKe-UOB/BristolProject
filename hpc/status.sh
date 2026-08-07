#!/bin/bash
# Where is the ladder up to? Prints the queue and the training progress of every checkpoint.
#
#   bash hpc/status.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"

if command -v squeue >/dev/null 2>&1; then
    echo "== queue"
    squeue -u "${USER}" -o "%.10i %.14j %.8T %.10M %.10L %R" || true
    echo
fi

echo "== prerequisites"
for f in artifacts/manifest_stats.json _imgcache.npy dino_clip_feats_v1.npz; do
    if [ -e "${f}" ]; then
        printf "  %-28s %s\n" "${f}" "$(du -h "${f}" 2>/dev/null | cut -f1)"
    else
        printf "  %-28s missing\n" "${f}"
    fi
done

echo
echo "== training progress (target 1000 steps each)"
python - <<'PY'
import glob, os, torch

groups = [
    ("Stage 1  CAP",       "_vitb_cap_s*_ckpt.pt",  5),
    ("Stage 3  holdout",   "_vitb_dst_s[5-9]_ckpt.pt", 5),
    ("Stage 3  deploy",    "_vitb_dep_s1[0-2]_ckpt.pt", 3),
    ("Stage 3  hard-CL",   "_vitb_hc2_s1[6-8]_ckpt.pt", 3),
    ("Stage 4  Mega",      "_vitb_mega_s4[0-2]_ckpt.pt", 3),
]
for title, pattern, expected in groups:
    found = sorted(glob.glob(pattern))
    print(f"  {title}  ({len(found)}/{expected} checkpoints)")
    for p in found:
        try:
            step = torch.load(p, map_location="cpu")["step"]
        except Exception as exc:
            print(f"     {os.path.basename(p):32s} unreadable ({exc.__class__.__name__})")
            continue
        bar = "#" * (step * 20 // 1000)
        print(f"     {os.path.basename(p):32s} {step:5d}/1000 |{bar:<20}|")
    if not found:
        print("     none yet")
PY

echo
echo "== artifacts2, most recently written first"
echo "   (these start as the archived reference results that came with the clone;"
echo "    your run replaces them as each stage finishes)"
ls -1t artifacts2/*.json 2>/dev/null | head -5 | sed 's/^/  /'
