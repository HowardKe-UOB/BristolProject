"""Final comprehensive fusion incl. the n_stage=2 Mega trio (mega2 s60/61/62).
Greedy forward selection (max P2) over the full zero-human pool + key hand combos,
each with optional cluster rerank. CPU, saved embeddings.

    python fuse_final.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json
import os
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
    ("_sweep_mega2_trio_emb.npz", ["mega2_60", "mega2_61", "mega2_62"]),
    ("_sweep_mega2ft_trio_emb.npz", ["m2ft80", "m2ft81", "m2ft82"]),
    ("_sweep_sup2_trio_emb.npz", ["sup2_90", "sup2_91", "sup2_92"]),
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
        # A fresh reproduction has only the core-ladder models; the extended pool
        # (megaft / mega2 / sup2 variants) may be absent. Skip loudly, never crash:
        # the greedy search then runs over whatever pool is present.
        if not os.path.exists(npz):
            print(f"[skip] {npz} not found -> pool loses {names}", flush=True)
            continue
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        for name, k in zip(names, [k for k in d.files if k != "ids"]):
            models[name] = d[k]
    if not models:
        raise SystemExit("no embedding files found; run eval_sweep.py on the trained "
                         "checkpoints first (see hpc/06b_embed_sweep.sbatch)")
    print(f"[pool] {len(models)} models: {sorted(models)}", flush=True)

    # protocol query/gallery sets
    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
    dorsal_qg = []
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gs = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gs]
        if q:
            dorsal_qg.append((q, g))

    dcache = {}

    def d1(name, q, g, tag):
        key = (name, tag)
        if key not in dcache:
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            dcache[key] = champ_dist(q, g, [models[name]], ids, cams_qg)
        return dcache[key]

    def cluster_labels(names):
        D = np.zeros((len(ids), len(ids)))
        for n in names:
            M = models[n]; Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            D += 1.0 - Mn @ Mn.T
        # NOTE: the 0.5 halves the mean cosine distance, so eps=0.35 below acts
        # like eps=0.7 on the full cosine scale -- a historical divergence from
        # fuse_hetero.py, which clusters the unhalved distance at eps=0.35. Kept
        # as-is: the archived numbers were produced this way; the thesis appendix
        # documents both variants.
        D = np.clip(0.5 * (D / len(names)), 0.0, None); np.fill_diagonal(D, 0.0)
        ip = {t: i for i, t in enumerate(ids)}
        for p in cl:
            a, b = tuple(p)
            if a in ip and b in ip:
                D[ip[a], ip[b]] = D[ip[b], ip[a]] = D.max()
        raw = DBSCAN(eps=0.35, min_samples=2, metric="precomputed").fit_predict(D)
        lab, nxt = {}, (int(raw.max()) + 1 if raw.max() >= 0 else 0)
        for t, l in zip(ids, raw):
            lab[t] = int(l) if l >= 0 else (nxt := nxt + 1)
        return lab

    def score(names, cluster=False):
        lab = cluster_labels(names) if cluster else None

        def fuse(q, g, tag):
            dd = np.mean([d1(n, q, g, tag) for n in names], axis=0)
            if lab is not None:
                la = np.array([lab[it.tracklet_id] for it in q])
                lb = np.array([lab[it.tracklet_id] for it in g])
                dd = dd - 0.05 * (la[:, None] == lb[None, :])
            return dd
        p1 = _score(q1, g1, fuse(q1, g1, "P1"), (1, 5, 10))
        p2 = _score(items, items, fuse(items, items, "P2"), (1, 5, 10))
        dors = [_score(q, g, fuse(q, g, f"d{i}"), (1, 5, 10))["rank-1"]
                for i, (q, g) in enumerate(dorsal_qg)]
        return p1, float(np.mean(dors)), p2

    # greedy forward selection maximizing P2
    sel, best, rem = [], -1, list(models)
    print("greedy (max P2):", flush=True)
    while rem:
        c = sorted(((score(sel + [m])[2]["rank-1"], m) for m in rem), reverse=True)
        p2v, m = c[0]
        if p2v <= best + 0.001 and len(sel) >= 3:
            break
        sel.append(m); rem.remove(m); best = p2v
        p1, dr, p2 = score(sel)
        print(f"  +{m:9s} set({len(sel)}) P1 {p1['rank-1']:.3f} | dorsal {dr:.3f} | "
              f"P2 {p2['rank-1']:.3f}", flush=True)

    report = {}
    combos = {
        "greedy-P2 set": sel,
        "sup2_trio": ["sup2_90", "sup2_91", "sup2_92"],
        "hc + all3 mega2 trios": ["hc16", "hc17", "hc18",
                                  "mega2_60", "mega2_61", "mega2_62",
                                  "m2ft80", "m2ft81", "m2ft82",
                                  "sup2_90", "sup2_91", "sup2_92"],
        "sup2_trio + mega2ft_trio": ["sup2_90", "sup2_91", "sup2_92",
                                     "m2ft80", "m2ft81", "m2ft82"],
        "all19": list(models),
    }
    print("\nkey combos (plain / +cluster):", flush=True)
    for name, names in combos.items():
        missing = [n for n in names if n not in models]
        if missing:
            print(f"  {name:34s}: skipped (missing {missing})", flush=True)
            continue
        for clu in (False, True):
            p1, dr, p2 = score(names, clu)
            k = name + ("+clust" if clu else "")
            print(f"  {k:34s}: P1 {p1['rank-1']:.3f} | dorsal {dr:.3f} | "
                  f"P2 {p2['rank-1']:.3f} (mAP {p2['mAP']:.3f})", flush=True)
            report[k] = {"P1": p1, "dorsal": dr, "P2": p2}
    with open("artifacts2/fuse_final_v1.json", "w", encoding="utf-8") as fh:
        json.dump({"greedy_set": sel, **report}, fh, indent=2)
    print("saved artifacts2/fuse_final_v1.json", flush=True)


if __name__ == "__main__":
    main()
