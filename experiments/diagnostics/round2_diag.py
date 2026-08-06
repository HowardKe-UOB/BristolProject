"""Round-2 teacher diagnostic: mutual-NN link precision in the 7-model ensemble
space (5 CAP + 2 distilled students) vs the round-1 5-model space. CPU-only.

    python round2_diag.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import numpy as np

from cowreid import Manifest, build_tracklets
from cowreid.tracklets import TrackletIndex
from consensus_ens import mutual_knn_links

HOLD = "66.130"


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    d5 = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    ids = list(d5["ids"]); pos = {t: i for i, t in enumerate(ids)}
    mats = [d5[s] for s in d5.files if s != "ids"]
    dst = np.load("_vitb_dst_emb_v2.npz", allow_pickle=True)
    ids2 = list(dst["ids"])
    order = [ids2.index(t) for t in ids]
    mats7 = mats + [dst[s][order] for s in dst.files if s != "ids"]

    g_tids = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    cams = [index.camera_of(t) for t in g_tids]
    rows = [pos[t] for t in g_tids]

    for name, ms in [("round1 (5 CAP)", mats), ("round2 (7 = 5 CAP + 2 students)", mats7)]:
        X = np.mean([m[rows] for m in ms], axis=0)
        X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        for k in (1, 2):
            links = mutual_knn_links(X.copy(), cams, k=k)
            pairs = [tuple(l) for l in links]
            correct = sum(gt[g_tids[a]] == gt[g_tids[b]] for a, b in pairs)
            print(f"  {name:34s} k={k}: {len(pairs):4d} links  "
                  f"precision={correct / max(len(pairs), 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
