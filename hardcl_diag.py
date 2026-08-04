"""Ammunition diagnostic for HARD CANNOT-LINK SHARPENING (CPU).

Cannot-link pairs (same-camera time-overlap, or non-overlapping-camera
time-overlap) are physically guaranteed different cows. The HARD ones -- pairs
whose current embeddings are highly similar -- are exactly the look-alike
discrimination signal that supervision normally provides. This script counts the
ammo in the current best feature space (k=2 trio ensemble):
  * similarity distribution of CL pairs, split by type (same-cam / cross-cam);
  * GT violation rate of CL pairs (should be ~0);
  * reference: similarity distribution of TRUE same-cow cross-camera pairs;
  * ammo table: #hard CL pairs above thresholds, and tracklet coverage.

    python hardcl_diag.py
"""
from __future__ import annotations

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import build_cannot_link
from cowreid.tracklets import TrackletIndex

K2 = ("s7", "s8", "s9")


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of

    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    X = np.mean([d[s] for s in sorted(d.files) if any(k in s for k in K2)], axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    have = set(ids)

    cl = build_cannot_link(tracklets, topo, 0.02)
    cl = [tuple(p) for p in cl if all(t in have for t in p)]
    print(f"{len(cl)} tracklet-level cannot-link pairs (embedded both sides)", flush=True)

    sims, types, viol = [], [], 0
    for a, b in cl:
        s = float(X[pos[a]] @ X[pos[b]])
        sims.append(s)
        types.append("same-cam" if cam_of(a) == cam_of(b) else "cross-cam")
        viol += (gt[a] == gt[b])
    sims = np.array(sims); types = np.array(types)
    print(f"GT violation rate of CL pairs: {viol}/{len(cl)} = {viol/len(cl):.4%}", flush=True)

    # positive reference: true same-cow cross-camera pairs
    from collections import defaultdict
    by_id = defaultdict(list)
    for t in ids:
        by_id[gt[t]].append(t)
    pos_sims = []
    for _g, ts in by_id.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                if cam_of(ts[i]) != cam_of(ts[j]):
                    pos_sims.append(float(X[pos[ts[i]]] @ X[pos[ts[j]]]))
    pos_sims = np.array(pos_sims)
    print(f"\nTRUE same-cow cross-cam pairs: n={len(pos_sims)}  "
          f"median={np.median(pos_sims):.3f}  p25={np.percentile(pos_sims,25):.3f}  "
          f"p75={np.percentile(pos_sims,75):.3f}", flush=True)

    for tp in ("same-cam", "cross-cam"):
        s = sims[types == tp]
        print(f"CL[{tp:9s}]: n={len(s):6d}  median={np.median(s):.3f}  "
              f"p90={np.percentile(s,90):.3f}  p99={np.percentile(s,99):.3f}  "
              f"max={s.max():.3f}", flush=True)

    print(f"\nAMMO TABLE  (hard CL pairs with cosine >= thr; positives median "
          f"= {np.median(pos_sims):.3f})", flush=True)
    print(f"  {'thr':>5s} {'same-cam':>9s} {'cross-cam':>10s} {'total':>7s} "
          f"{'tracklets covered':>18s} {'dorsal-only pairs':>18s}", flush=True)
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        m = sims >= thr
        cov = set()
        n_dorsal = 0
        for (a, b), hit in zip(cl, m):
            if hit:
                cov.add(a); cov.add(b)
                if cam_of(a) != "66.130" and cam_of(b) != "66.130":
                    n_dorsal += 1
        print(f"  {thr:5.1f} {int((m & (types=='same-cam')).sum()):9d} "
              f"{int((m & (types=='cross-cam')).sum()):10d} {int(m.sum()):7d} "
              f"{len(cov):18d} {n_dorsal:18d}", flush=True)

    # violation rate among HARD pairs specifically (the ones we would train on)
    for thr in (0.5, 0.6, 0.7):
        m = sims >= thr
        v = sum(gt[a] == gt[b] for (a, b), hit in zip(cl, m) if hit)
        print(f"  violation rate at thr {thr}: {v}/{int(m.sum())}"
              f" = {v/max(int(m.sum()),1):.3%}", flush=True)


if __name__ == "__main__":
    main()
