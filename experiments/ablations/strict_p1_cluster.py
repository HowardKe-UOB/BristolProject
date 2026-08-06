"""Cluster-consistency rerank on the STRICT P1 benchmark (holdout trio).

    python strict_p1_cluster.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    by_tid = {t.tracklet_id: t for t in tracklets}

    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"])
    embs = [d[k] for k in sorted(d.files)
            if any(s in k for s in ("s7", "s8", "s9"))]
    Xf = np.mean([M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                  for M in embs], axis=0)
    Xf = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)
    cams_all = [by_tid[t].camera for t in ids]
    lab = dbscan_cluster(ids, Xf, cams_all, eps=0.5, cannot_link=cl)
    print(f"clusters: {num_clusters(lab)}", flush=True)

    g1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL
          and by_tid[t].gt_label in {x.identity for x in g1}]
    cams_qg = [it.camera for it in q1] + [it.camera for it in g1]
    base = champ_dist(q1, g1, embs, ids, cams_qg)
    la = np.array([lab[it.tracklet_id] for it in q1])
    lb = np.array([lab[it.tracklet_id] for it in g1])
    same = la[:, None] == lb[None, :]
    r0 = _score(q1, g1, base, (1, 5, 10))
    print(f"strict P1 base     : r1={r0['rank-1']:.3f} r5={r0['rank-5']:.3f} "
          f"mAP={r0['mAP']:.3f}", flush=True)
    for bonus in (0.05, 0.15):
        r1 = _score(q1, g1, base - bonus * same, (1, 5, 10))
        print(f"strict P1 +cluster (bonus {bonus}): r1={r1['rank-1']:.3f} "
              f"r5={r1['rank-5']:.3f} mAP={r1['mAP']:.3f}", flush=True)


if __name__ == "__main__":
    main()
