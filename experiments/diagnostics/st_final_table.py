"""Consolidated final comparison: supervised vs unsupervised ViT-B (SAME checkpoints,
SAME label-free inference levers), scored identically in one place from the saved
embedding npz files. CPU-only, no GPU, no training -- safe to re-run.

    python st_final_table.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from st_validate2 import run_all

HOLD = "66.130"


def load_emb(npz, key):
    d = np.load(npz, allow_pickle=True)
    ids = list(d["ids"])
    return {t: v for t, v in zip(ids, d[key])}


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]

    report = {}
    for mode, npz in [("UNSUPERVISED", "_vitb_unsup_emb_v1.npz"),
                      ("SUPERVISED", "_vitb_sup_emb_v1.npz")]:
        for key in ("emb256", "feat768"):
            emb = load_emb(npz, key)
            tag = f"{mode}/{key}/"
            print(f"\n=== {mode}  {key}  (leave-out {HOLD}) ===", flush=True)
            run_all(query, gallery, emb, index, topo, margin=0, tag=tag, report=report)

    # headline extraction
    def g(tag, variant, m):
        return report[tag][variant][m]

    print("\n" + "=" * 72)
    print("HEADLINE (rank-1 / rank-5 / mAP), leave-out 66.130")
    print("=" * 72)
    rows = [
        ("UNSUP  emb256  cosine       (old protocol)", "UNSUPERVISED/emb256/", "cosine"),
        ("UNSUP  feat768 cosine       (backbone feat)", "UNSUPERVISED/feat768/", "cosine"),
        ("UNSUP  feat768 CC           (+cam-center)  ", "UNSUPERVISED/feat768/", "CC"),
        ("UNSUP  feat768 CC+RR        (+re-rank)     ", "UNSUPERVISED/feat768/", "CC+RR"),
        ("UNSUP  feat768 CC+RR+ST     (+ST mask)     ", "UNSUPERVISED/feat768/", "CC+RR+ST"),
        ("SUP    emb256  cosine       (old protocol)", "SUPERVISED/emb256/", "cosine"),
        ("SUP    feat768 cosine       (backbone feat)", "SUPERVISED/feat768/", "cosine"),
        ("SUP    feat768 cosine+RR    (+re-rank)     ", "SUPERVISED/feat768/", "cosine+RR"),
        ("SUP    feat768 cosine+RR+ST (+ST mask)     ", "SUPERVISED/feat768/", "cosine+RR+ST"),
    ]
    for label, tag, variant in rows:
        print(f"  {label}: r1={g(tag,variant,'rank-1'):.3f}  "
              f"r5={g(tag,variant,'rank-5'):.3f}  mAP={g(tag,variant,'mAP'):.3f}")

    with open("artifacts2/st_final_comparison_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nsaved artifacts2/st_final_comparison_v1.json")


if __name__ == "__main__":
    main()
