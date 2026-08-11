"""Ensemble-size curve for the CAP seeds (CPU-only, from saved 5-seed embeddings).

For k = 1..5, evaluate the distance-mean ensemble over EVERY C(5,k) seed subset
and report mean +/- std of rank-1 / rank-5 / mAP. This quantifies how many seeds
the ensemble needs to stabilise, without any seed selection.

    python cap_ens_curve.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import itertools
import json

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

HOLD = "66.130"
RANKS = (1, 5, 10)


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
    q, g = list(query), list(gallery)
    cams_qg = [it.camera for it in q] + [it.camera for it in g]
    mask = build_st_mask(q, g, index, topo, margin=0)

    d = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    seeds = [k for k in d.files if k != "ids"]

    def champ_dist(M):
        E = {t: M[pos[t]] for t in ids}
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    print("computing per-seed champion distances...", flush=True)
    dists = {s: champ_dist(d[s]) for s in seeds}

    out = {}
    print(f"\nensemble-size curve (distance-mean over seed subsets, +ST mask):", flush=True)
    print(f"  k   rank-1          rank-5          mAP             (n subsets)", flush=True)
    for k in range(1, len(seeds) + 1):
        r1s, r5s, maps = [], [], []
        for sub in itertools.combinations(seeds, k):
            dm = np.mean([dists[s] for s in sub], axis=0)
            dm = dm.copy(); dm[mask] = INF
            r = _score(q, g, dm, RANKS)
            r1s.append(r["rank-1"]); r5s.append(r["rank-5"]); maps.append(r["mAP"])
        r1s, r5s, maps = map(np.array, (r1s, r5s, maps))
        out[k] = {"rank1_mean": float(r1s.mean()), "rank1_std": float(r1s.std()),
                  "rank5_mean": float(r5s.mean()), "rank5_std": float(r5s.std()),
                  "mAP_mean": float(maps.mean()), "mAP_std": float(maps.std()),
                  "n_subsets": len(r1s),
                  "rank1_min": float(r1s.min()), "rank1_max": float(r1s.max())}
        print(f"  {k}   {r1s.mean():.3f}+/-{r1s.std():.3f}   {r5s.mean():.3f}+/-{r5s.std():.3f}"
              f"   {maps.mean():.3f}+/-{maps.std():.3f}   ({len(r1s)})"
              f"   [r1 min {r1s.min():.3f} max {r1s.max():.3f}]", flush=True)

    with open("artifacts2/cap_ens_curve_v1.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved artifacts2/cap_ens_curve_v1.json")


if __name__ == "__main__":
    main()
