"""Final zero-human numbers: the greedy-selected 4-model set {hc16, mega40, hc18,
megaft50} + all-10, each with and without cluster rerank on fused cosine distance.
Also the max-P1 view. CPU, saved embeddings.

    python final_best.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"
SRC = [
    ("_sweep_final_zerohuman_emb.npz", ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18"]),
    ("_sweep_mega_trio_emb.npz", ["mega40", "mega41", "mega42"]),
    ("_sweep_megaft_s50_emb.npz", ["megaft50"]),
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ids = None; models = {}
    for npz, names in SRC:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        for name, k in zip(names, [k for k in d.files if k != "ids"]):
            models[name] = d[k]

    def cluster_labels(embs):
        D = np.zeros((len(ids), len(ids)))
        for M in embs:
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            D += 1.0 - Mn @ Mn.T
        D = np.clip(0.5 * (D / len(embs)), 0.0, None)
        D = 0.5 * (D + D.T); np.fill_diagonal(D, 0.0)
        ipos = {t: i for i, t in enumerate(ids)}
        for p in cl:
            a, b = tuple(p)
            if a in ipos and b in ipos:
                D[ipos[a], ipos[b]] = D[ipos[b], ipos[a]] = D.max()
        raw = DBSCAN(eps=0.35, min_samples=2, metric="precomputed").fit_predict(D)
        lab, nxt = {}, (int(raw.max()) + 1 if raw.max() >= 0 else 0)
        for t, l in zip(ids, raw):
            if l >= 0:
                lab[t] = int(l)
            else:
                lab[t] = nxt; nxt += 1
        return lab

    def evalset(names, cluster):
        embs = [models[n] for n in names]
        lab = cluster_labels(embs) if cluster else None

        def dist(q, g):
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            dd = champ_dist(q, g, embs, ids, cams_qg)
            if lab is not None:
                la = np.array([lab[it.tracklet_id] for it in q])
                lb = np.array([lab[it.tracklet_id] for it in g])
                dd = dd - 0.05 * (la[:, None] == lb[None, :])
            return dd

        g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
        q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
              if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
        p1 = _score(q1, g1, dist(q1, g1), (1, 5, 10))
        dm = []
        for X in sorted({by_tid[t].camera for t in ids}):
            if X == OBL:
                continue
            g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
            gset = {it.identity for it in g}
            q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
                 if by_tid[t].camera == X and gt[t] in gset]
            if q:
                dm.append(_score(q, g, dist(q, g), (1, 5, 10))["rank-1"])
        id_cams = defaultdict(set)
        for t in ids:
            id_cams[gt[t]].add(by_tid[t].camera)
        multi = {i for i, c in id_cams.items() if len(c) >= 2}
        items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
        p2 = _score(items, items, dist(items, items), (1, 5, 10))
        return p1, float(np.mean(dm)), p2

    best4 = ["hc16", "mega40", "hc18", "megaft50"]
    all10 = list(models)
    report = {}
    for tag, names in [("best4", best4), ("all10", all10)]:
        for cluster in (False, True):
            p1, dr, p2 = evalset(names, cluster)
            key = f"{tag}{'+clust' if cluster else ''}"
            print(f"{key:16s}: P1 r1={p1['rank-1']:.3f} r5={p1['rank-5']:.3f} "
                  f"mAP={p1['mAP']:.3f} | dorsal {dr:.3f} | P2 r1={p2['rank-1']:.3f} "
                  f"r5={p2['rank-5']:.3f} mAP={p2['mAP']:.3f}", flush=True)
            report[key] = {"P1": p1, "dorsal_mean": dr, "P2": p2}

    with open("artifacts2/final_best_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/final_best_v1.json", flush=True)


if __name__ == "__main__":
    main()
