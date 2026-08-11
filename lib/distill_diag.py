"""Diagnostic: how good are pseudo-labels built in the 5-seed ENSEMBLE space?
(CPU-only; GT used to MEASURE quality only.)

Clusterings tested on the 834 train-camera tracklets:
  * CA-Jaccard DBSCAN at several eps (the training pipeline's method);
  * agglomerative (average linkage) on cosine distance with n_clusters set from
    the HERD-SIZE prior (the farm knows roughly how many cows are in the barn --
    operational metadata, not identity labels).
Metrics: #clusters, pairwise precision/recall/F1 vs GT, cross-camera pair recall
(the part that matters for the bottleneck).

    python distill_diag.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import build_cannot_link
from cowreid.tracklets import TrackletIndex

HOLD = "66.130"


def pair_metrics(labels, gt, cam_of):
    """Pairwise precision/recall over all tracklet pairs, plus cross-camera-only
    recall/precision."""
    tids = list(labels)
    by_lab = defaultdict(list)
    for t in tids:
        by_lab[labels[t]].append(t)
    tp = fp = tp_cc = fp_cc = 0
    for members in by_lab.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                same = gt[a] == gt[b]
                cc = cam_of(a) != cam_of(b)
                tp += same; fp += (not same)
                if cc:
                    tp_cc += same; fp_cc += (not same)
    # totals for recall
    tot = tot_cc = 0
    by_gt = defaultdict(list)
    for t in tids:
        by_gt[gt[t]].append(t)
    for members in by_gt.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                tot += 1
                if cam_of(members[i]) != cam_of(members[j]):
                    tot_cc += 1
    prec = tp / max(tp + fp, 1); rec = tp / max(tot, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    prec_cc = tp_cc / max(tp_cc + fp_cc, 1); rec_cc = tp_cc / max(tot_cc, 1)
    return {"precision": round(prec, 3), "recall": round(rec, 3), "F1": round(f1, 3),
            "cc_precision": round(prec_cc, 3), "cc_recall": round(rec_cc, 3),
            "cc_pairs_made": tp_cc + fp_cc}


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    d = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    seeds = [k for k in d.files if k != "ids"]
    Xens = np.mean([d[s] for s in seeds], axis=0)
    Xens = Xens / (np.linalg.norm(Xens, axis=1, keepdims=True) + 1e-12)

    g_tids = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    X = np.stack([Xens[pos[t]] for t in g_tids])
    cams = [index.camera_of(t) for t in g_tids]
    n_ids_true = len({gt[t] for t in g_tids})
    print(f"{len(g_tids)} train tracklets, {n_ids_true} true identities\n", flush=True)

    report = {}
    print("-- CA-Jaccard DBSCAN on ENSEMBLE space --", flush=True)
    for eps in (0.4, 0.5, 0.6):
        lab = dbscan_cluster(g_tids, X, cams, eps=eps, cannot_link=cl)
        m = pair_metrics(lab, gt, index.camera_of)
        m["n_clusters"] = num_clusters(lab)
        report[f"dbscan_eps{eps}"] = m
        print(f"  eps={eps}: clusters={m['n_clusters']:4d}  P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['F1']:.3f}  | cross-cam P={m['cc_precision']:.3f} "
              f"R={m['cc_recall']:.3f} ({m['cc_pairs_made']} pairs)", flush=True)

    print("\n-- agglomerative (avg cosine) with herd-size prior --", flush=True)
    from sklearn.cluster import AgglomerativeClustering
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    for K in (81, 98, 120):
        ac = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average")
        arr = ac.fit_predict(D)
        lab = {t: int(l) for t, l in zip(g_tids, arr)}
        m = pair_metrics(lab, gt, index.camera_of)
        m["n_clusters"] = K
        report[f"agglo_K{K}"] = m
        print(f"  K={K}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['F1']:.3f}"
              f"  | cross-cam P={m['cc_precision']:.3f} R={m['cc_recall']:.3f} "
              f"({m['cc_pairs_made']} pairs)", flush=True)

    print("\n-- reference: single-seed s0 space, same clusterings --", flush=True)
    X0 = d["s0"]; X0 = np.stack([X0[pos[t]] for t in g_tids])
    X0 = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-12)
    lab = dbscan_cluster(g_tids, X0, cams, eps=0.5, cannot_link=cl)
    m = pair_metrics(lab, gt, index.camera_of); m["n_clusters"] = num_clusters(lab)
    report["s0_dbscan_eps0.5"] = m
    print(f"  s0 eps=0.5: clusters={m['n_clusters']:4d}  P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F1={m['F1']:.3f}  | cc P={m['cc_precision']:.3f} "
          f"R={m['cc_recall']:.3f}", flush=True)
    D0 = 1.0 - X0 @ X0.T; np.fill_diagonal(D0, 0.0)
    ac = AgglomerativeClustering(n_clusters=98, metric="precomputed", linkage="average")
    lab = {t: int(l) for t, l in zip(g_tids, ac.fit_predict(D0))}
    m = pair_metrics(lab, gt, index.camera_of); m["n_clusters"] = 98
    report["s0_agglo_K98"] = m
    print(f"  s0 K=98 : P={m['precision']:.3f} R={m['recall']:.3f} F1={m['F1']:.3f}"
          f"  | cc P={m['cc_precision']:.3f} R={m['cc_recall']:.3f}", flush=True)

    with open("artifacts2/distill_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\n(training-time reference: crop-OT links ~15% precision; "
          "mid-run DBSCAN collapsed to 25-60 clusters)")
    print("saved artifacts2/distill_diag_v1.json")


if __name__ == "__main__":
    main()
