"""Heterogeneous fusion: combine DINOv2 and MegaDescriptor student embeddings
(saved _sweep_*_emb.npz) at the distance level, across all protocols + cluster
rerank. CPU only. Tests whether the animal-ReID backbone is COMPLEMENTARY to
DINOv2 (dorsal-strong + oblique-strong -> both up).

    python fuse_hetero.py
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
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"
SOURCES = {
    "dino6": "_sweep_final_zerohuman_emb.npz",   # dep s10-12 + hc2 s16-18
    "mega3": "_sweep_mega_trio_emb.npz",         # mega s40-42
    "act": "_sweep_act_stack_emb.npz",           # dep3 + act n300/n1000 (1000 clicks)
}


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return list(d["ids"]), [d[k] for k in d.files if k != "ids"]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    banks = {}
    ref_ids = None
    for name, npz in SOURCES.items():
        ids, ms = load(npz)
        if ref_ids is None:
            ref_ids = ids
        assert ids == ref_ids, f"{name} id order mismatch"
        banks[name] = ms
    ids = ref_ids
    print({k: len(v) for k, v in banks.items()}, flush=True)

    combos = {
        "dino6 (zero-human ref)": banks["dino6"],
        "mega3": banks["mega3"],
        "dino6+mega3": banks["dino6"] + banks["mega3"],
        "dino6+mega3 (mega x2)": banks["dino6"] + banks["mega3"] * 2,
        "dino6+mega3+act(clicks)": banks["dino6"] + banks["mega3"] + banks["act"],
    }

    def eval_all(embs, name, cluster=False):
        lab = None
        if cluster:
            # heterogeneous dims -> cluster on the FUSED COSINE distance (mean of
            # per-model cosine distances; non-negative, scale-consistent).
            from sklearn.cluster import DBSCAN
            D = np.zeros((len(ids), len(ids)), dtype=np.float64)
            for M in embs:
                Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                D += 1.0 - Mn @ Mn.T
            D /= len(embs)
            D = np.clip(0.5 * (D + D.T), 0.0, None); np.fill_diagonal(D, 0.0)
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
        tag = name + (" +clust" if cluster else "")
        print(f"  {tag:32s}: P1 {p1['rank-1']:.3f} | dorsal {np.mean(dm):.3f} | "
              f"P2 {p2['rank-1']:.3f} (r5 {p2['rank-5']:.3f} mAP {p2['mAP']:.3f})",
              flush=True)
        return {"P1": p1, "dorsal": float(np.mean(dm)), "P2": p2}

    report = {}
    for name, embs in combos.items():
        report[name] = eval_all(embs, name, cluster=False)
    print("  --- with cluster rerank (on fused distance) ---", flush=True)
    for name in ("dino6+mega3", "dino6+mega3+act(clicks)"):
        report[name + " +clust"] = eval_all(combos[name], name, cluster=True)

    with open("artifacts2/fuse_hetero_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/fuse_hetero_v1.json", flush=True)


if __name__ == "__main__":
    main()
