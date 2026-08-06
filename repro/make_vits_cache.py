"""Build the frozen ViT-S feature cache that Stage 1 reads (dino_clip_feats_v1.npz).

Stage 1 (`repro/vitb_unsup_cap.py`) loads this cache unconditionally: it holds, for each
of the 1,079 tracklets, the 8 sampled frames' frozen DINOv2 ViT-S/14 features (8 x 384),
used to seed the camera-aware proxies before any training has happened.

The cache was originally produced as a side effect of the superseded phase-2 driver.
This script exposes the same routine on its own so a fresh clone can build it in one
step. It calls the identical function, so the cache is byte-for-byte what the reported
runs used. If the file already exists it is loaded, never overwritten.

    python repro/make_vits_cache.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz

Roughly 10 minutes on one GPU (it extracts and embeds ~8,600 crops), then ~9 MB on disk.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import os

import torch

from cowreid import Manifest
from cowreid.tracklets import build_tracklets
from train_phase2_run import cache_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--tar", default="2025Sep18.tar.gz")
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU visible; feature extraction will be slow")

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    print(f"[cache] {len(tracklets)} tracklets, {args.frames} frames each -> {args.cache}")

    existed = os.path.exists(args.cache)
    clips = cache_features(tracklets, args.tar, args.work, args.cache, args.frames, device)

    any_clip = next(iter(clips.values()))
    print(f"[cache] {'loaded existing' if existed else 'wrote'} {args.cache}: "
          f"{len(clips)} tracklets, feature shape {any_clip.shape}")


if __name__ == "__main__":
    main()
