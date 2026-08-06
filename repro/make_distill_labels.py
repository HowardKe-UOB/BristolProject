"""Build & validate DISTILLATION pseudo-labels (CPU).

Design: intra-camera clusters (easy, reliable) merged across cameras ONLY via
ensemble-space mutual-kNN links (60%/52% precision at k=1/k=2) -- no cross-camera
DBSCAN (whose transitive merges collapse precision to ~20%). Labels are FIXED
(teacher space), so the student cannot drift/over-merge them. GT used to MEASURE
quality only. Saves the chosen label set to a json for the student trainer.

    python make_distill_labels.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.tracklets import TrackletIndex
from consensus_ens import mutual_knn_links
from distill_diag import pair_metrics

HOLD = "66.130"
OUT = "artifacts2/distill_labels_v1.json"


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
    X = np.stack([Xens[pos[t]] for t in g_tids]).astype(np.float32)
    cams = [index.camera_of(t) for t in g_tids]

    # intra-camera clusters on ENSEMBLE embeddings (per camera, cannot-link-aware)
    cl_same = {p for p in cl if len({index.camera_of(t) for t in p}) == 1}
    by_cam = defaultdict(list)
    for t in g_tids:
        by_cam[index.camera_of(t)].append(t)
    intra = {}
    off = 0
    for c, ts in sorted(by_cam.items()):
        E = np.stack([Xens[pos[t]] for t in ts])
        lab = ClusterAssigner(0.7, 10).assign(ts, E, cl_same)
        for t in ts:
            intra[t] = off + lab[t]
        off += ClusterAssigner.num_clusters(lab)
    m0 = pair_metrics(intra, gt, index.camera_of)
    print(f"intra-only: clusters={len(set(intra.values()))}  P={m0['precision']:.3f} "
          f"R={m0['recall']:.3f}  (cc pairs made: {m0['cc_pairs_made']})", flush=True)

    def union_labels(base, links_idx):
        parent = dict(base)
        lab2root = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        # represent each intra cluster by an anchor tid
        anchor = {}
        for t, l in base.items():
            anchor.setdefault(l, t)
        parent = {t: anchor[base[t]] for t in base}
        for t in parent.values():
            parent[t] = t
        for a, b in links_idx:
            ta, tb = g_tids[a], g_tids[b]
            ra, rb = find(anchor[base[ta]]), find(anchor[base[tb]])
            if ra != rb:
                parent[rb] = ra
        roots = {}
        out = {}
        for t in base:
            r = find(anchor[base[t]])
            if r not in roots:
                roots[r] = len(roots)
            out[t] = roots[r]
        return out

    report = {"intra_only": {**m0, "n_clusters": len(set(intra.values()))}}
    best = None
    for k in (1, 2):
        links = mutual_knn_links(X.copy(), cams, k=k)
        links_idx = [tuple(l) for l in links]
        correct = sum(gt[g_tids[a]] == gt[g_tids[b]] for a, b in links_idx)
        lab = union_labels(intra, links_idx)
        m = pair_metrics(lab, gt, index.camera_of)
        nc = len(set(lab.values()))
        report[f"merged_k{k}"] = {**m, "n_clusters": nc, "links": len(links_idx),
                                  "link_precision": round(correct / max(len(links_idx), 1), 3)}
        print(f"merged k={k}: links={len(links_idx)} (link-P={correct/len(links_idx):.3f})  "
              f"clusters={nc}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['F1']:.3f}  "
              f"| cc P={m['cc_precision']:.3f} R={m['cc_recall']:.3f} "
              f"({m['cc_pairs_made']} pairs)", flush=True)
        if k == 1:
            best = lab

    json.dump({"labels": best, "note": "intra(ensemble)+mutualNN k=1 merge"},
              open(OUT, "w", encoding="utf-8"))
    print(f"\nsaved k=1 label set -> {OUT}", flush=True)
    with open("artifacts2/distill_labels_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
