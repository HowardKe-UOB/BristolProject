"""Guard check for cluster-consistency reranking: apply the P2-winning config
(eps=0.5, bonus=0.05 on the 4-model fused space) to the per-camera sweep and P1,
to confirm it generalizes across protocols and does not hurt.

    python cluster_rerank_guard.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"
EPS, BONUS = 0.5, 0.05


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="_sweep_dep3_hc2_emb.npz")
    ap.add_argument("--out", default="artifacts2/cluster_rerank_guard_v1.json")
    args = ap.parse_args()

    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    by_tid = {t.tracklet_id: t for t in tracklets}

    ds = np.load(args.npz, allow_pickle=True)
    sids = list(ds["ids"])
    embs = [ds[k] for k in ds.files if k != "ids"]
    Xf = np.mean([M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                  for M in embs], axis=0)
    Xf = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)

    # one global clustering over ALL tracklets (label-free, reused per protocol)
    cams_all = [by_tid[t].camera for t in sids]
    lab = dbscan_cluster(sids, Xf, cams_all, eps=EPS, cannot_link=cl)
    print(f"global clustering: {num_clusters(lab)} clusters (81 true IDs in train)",
          flush=True)
    arr = {t: lab[t] for t in sids}

    def run(q, g, name, report):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        base = champ_dist(q, g, embs, sids, cams_qg)
        la = np.array([arr[it.tracklet_id] for it in q])
        lb = np.array([arr[it.tracklet_id] for it in g])
        same = la[:, None] == lb[None, :]
        boosted = base - BONUS * same
        r0 = _score(q, g, base, (1, 5, 10))
        r1 = _score(q, g, boosted, (1, 5, 10))
        print(f"  {name:12s}: base r1={r0['rank-1']:.3f} -> +cluster "
              f"r1={r1['rank-1']:.3f}   (r5 {r0['rank-5']:.3f}->{r1['rank-5']:.3f}, "
              f"mAP {r0['mAP']:.3f}->{r1['mAP']:.3f})", flush=True)
        report[name] = {"base": r0, "clustered": r1}
        return r1

    report = {}
    # P1
    g1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
          if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
          if by_tid[t].camera == OBL
          and by_tid[t].gt_label in {x.identity for x in g1}]
    run(q1, g1, "P1", report)
    # per-camera sweep
    dorsal = []
    for X in sorted({by_tid[t].camera for t in sids}):
        g = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
             if by_tid[t].camera != X]
        gset = {it.identity for it in g}
        q = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
             if by_tid[t].camera == X and by_tid[t].gt_label in gset]
        if not q:
            continue
        r = run(q, g, f"query_{X}", report)
        if X != OBL:
            dorsal.append(r["rank-1"])
    print(f"  dorsal mean (clustered): {np.mean(dorsal):.3f}  (base was 0.538)",
          flush=True)
    # P2
    id_cams = defaultdict(set)
    for t in sids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
             if by_tid[t].gt_label in multi]
    run(items, items, "P2", report)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
