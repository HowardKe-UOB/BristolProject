"""Diagnostics for the two remaining CPU-testable levers (no training).

(A) CROSS-STUDENT LINK AGREEMENT: mine mutual-kNN cross-camera links separately
    in each student's space (s7/s8/s9, all 7 cameras) and check whether links
    agreed by >=2 or all 3 students have higher precision than trio-space links
    (352 @ 56%). If agreement filtering reaches ~75%+ at usable counts, a student
    retrained on agreed links could be better.

(B) CLUSTER-CONSISTENCY RERANK on P2: cluster the current best 4-model fused
    space (constrained, several granularities), then subtract a bonus from
    same-cluster pair distances and re-score P2. Tests whether global structure
    adds anything over pairwise retrieval.

    python next_levers_diag.py
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
from consensus_ens import mutual_knn_links
from eval_sweep import champ_dist

OBL = "66.130"


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of
    report = {}

    # ---------------- (A) cross-student link agreement ---------------- #
    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"])
    cams_list = [cam_of(t) for t in ids]
    print("[A] per-student mutual-2NN links over all 7 cameras:", flush=True)
    per_student = []
    for s in ("s7", "s8", "s9"):
        key = [k for k in d.files if s in k][0]
        X = d[key]
        X = (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
        links = mutual_knn_links(X.copy(), cams_list, k=2)
        pairs = {frozenset((ids[a], ids[b])) for a, b in (tuple(l) for l in links)}
        prec = np.mean([gt[tuple(p)[0]] == gt[tuple(p)[1]] for p in pairs])
        print(f"    {s}: {len(pairs)} links  precision={prec:.3f}", flush=True)
        per_student.append(pairs)
    from collections import Counter
    cnt = Counter()
    for pairs in per_student:
        for p in pairs:
            cnt[p] += 1
    for need in (2, 3):
        agreed = [p for p, c in cnt.items() if c >= need]
        prec = np.mean([gt[tuple(p)[0]] == gt[tuple(p)[1]] for p in agreed])
        n_obl = sum(1 for p in agreed
                    if OBL in (cam_of(tuple(p)[0]), cam_of(tuple(p)[1])))
        print(f"    agreed by >= {need}: {len(agreed)} links  precision={prec:.3f}  "
              f"({n_obl} involve {OBL})", flush=True)
        report[f"agree_{need}"] = {"links": len(agreed), "precision": round(float(prec), 3)}

    # ---------------- (B) cluster-consistency rerank on P2 ---------------- #
    ds = np.load("_sweep_dep3_hc2_emb.npz", allow_pickle=True)
    sids = list(ds["ids"])
    embs = [ds[k] for k in ds.files if k != "ids"]
    by_tid = {t.tracklet_id: t for t in tracklets}
    id_cams = defaultdict(set)
    for t in sids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in sids
             if by_tid[t].gt_label in multi]
    cams2 = [it.camera for it in items] * 2
    base = champ_dist(items, items, embs, sids, cams2)
    r0 = _score(items, items, base, (1, 5, 10))
    print(f"\n[B] P2 baseline: r1={r0['rank-1']:.3f} r5={r0['rank-5']:.3f} "
          f"mAP={r0['mAP']:.3f}", flush=True)

    Xf = np.mean([M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                  for M in embs], axis=0)
    Xf = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-12)
    item_tids = [it.tracklet_id for it in items]
    rows = [sids.index(t) for t in item_tids]
    Xi = Xf[rows]
    cams_i = [it.camera for it in items]
    for eps in (0.45, 0.5, 0.55):
        lab = dbscan_cluster(item_tids, Xi, cams_i, eps=eps, cannot_link=cl)
        nc = num_clusters(lab)
        same = np.zeros_like(base, dtype=bool)
        arr = np.array([lab[t] for t in item_tids])
        same = arr[:, None] == arr[None, :]
        for bonus in (0.05, 0.15):
            dd = base - bonus * same
            r = _score(items, items, dd, (1, 5, 10))
            print(f"    eps={eps} (#c={nc}) bonus={bonus}: r1={r['rank-1']:.3f} "
                  f"r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}", flush=True)
            report[f"cluster_eps{eps}_b{bonus}"] = r

    with open("artifacts2/next_levers_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/next_levers_diag_v1.json", flush=True)


if __name__ == "__main__":
    main()
