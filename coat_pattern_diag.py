"""Coat-pattern visibility diagnostic (archives the basis for ch3's 'fragmentary
appearance' claim).

For a seeded random sample of crops, classify each single crop by how much of
the coat's two colours it exposes:
  - bicolor   : both bright and dark coat regions visible (>=15% each of the
                foreground) -> the patch pattern is (partly) visible in this crop
  - mostly-dark / mostly-light : the crop exposes essentially one colour patch

Interpretation: the herd is bicolor-patched, but roughly half of single
top-down crops expose only one colour region -- the pattern exists at animal
level yet is only fragmentarily visible per view. Heuristic, foreground =
pixels that differ from the flat grey matte background.

    python coat_pattern_diag.py            # n=400, seed=1 (the cited setting)
    python coat_pattern_diag.py --n 1000 --seed 2
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile

import numpy as np
from PIL import Image

TAR = "2025Sep18.tar.gz"
OUT = "artifacts2/coat_pattern_diag_v1.json"


def classify(img: Image.Image) -> str | None:
    a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    sat = a.max(axis=2) - a.min(axis=2)
    v = a.mean(axis=2)
    fg = (np.abs(v - 0.5) > 0.12) | (sat > 0.08)   # not the flat grey matte
    if fg.sum() < 200:
        return None
    vv = v[fg]
    bright = float((vv > 0.55).mean())
    dark = float((vv < 0.35).mean())
    if bright > 0.15 and dark > 0.15:
        return "bicolor"
    return "mostly_dark" if dark >= bright else "mostly_light"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out_path = OUT
    if os.path.exists(out_path):                       # never overwrite artifacts
        i = 2
        while os.path.exists(out_path := OUT.replace("_v1", f"_v{i}")):
            i += 1

    tar = tarfile.open(TAR, "r:gz")
    names = [m.name for m in tar if m.name.endswith(".jpeg")]
    random.seed(args.seed)
    picks = random.sample(names, args.n)

    counts = {"bicolor": 0, "mostly_dark": 0, "mostly_light": 0}
    per_camera: dict[str, dict[str, int]] = {}
    skipped = 0
    for name in picks:
        cls = classify(Image.open(io.BytesIO(tar.extractfile(name).read())))
        if cls is None:
            skipped += 1
            continue
        counts[cls] += 1
        cam = name.split("/")[1]
        per_camera.setdefault(cam, {"bicolor": 0, "mostly_dark": 0, "mostly_light": 0})[cls] += 1

    tot = sum(counts.values())
    report = {
        "script": "coat_pattern_diag.py",
        "tar": TAR,
        "n_sampled": args.n,
        "seed": args.seed,
        "n_classified": tot,
        "n_skipped_tiny_foreground": skipped,
        "counts": counts,
        "fractions": {k: round(c / tot, 4) for k, c in counts.items()},
        "per_camera": per_camera,
        "heuristic": "fg = |V-0.5|>0.12 or sat>0.08 on 64x64; bicolor iff "
                     ">15% of fg bright (V>0.55) AND >15% dark (V<0.35)",
        "note": "single-crop colour exposure; herd is bicolor-patched at animal "
                "level, but ~half of single crops expose one colour patch only",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"{tot} classified: " + ", ".join(f"{k} {v} ({v/tot:.0%})" for k, v in counts.items()))
    print("saved", out_path)


if __name__ == "__main__":
    main()
