"""Greedy ensemble-composition search over ALL zero-human student embeddings
(CPU, saved sweep npzs). Finds the best distance-fusion subset for P2, reporting
dorsal and P1 for the chosen set, with and without cluster rerank.

Pool (distinct zero-human models):
  dino-deploy s10/s11/s12, dino-hc2 s16/s17/s18  (from final_zerohuman npz)
  mega s40/s41/s42                                (from mega_trio npz)
  megaft s50                                      (from megaft_s50 npz)
(act n300/n1000 excluded -- they use clicks.)

    python ensemble_search.py
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
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"
POOL = [
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

    ref_ids = None
    models = {}
    for npz, names in POOL:
        d = np.load(npz, allow_pickle=True)
        ids = list(d["ids"])
        if ref_ids is None:
            ref_ids = ids
        assert ids == ref_ids
        mkeys = [k for k in d.files if k != "ids"]
        for name, k in zip(names, mkeys):
            models[name] = d[k]
    ids = ref_ids
    print(f"pool: {list(models)}", flush=True)

    # precompute per-model champion distance for each protocol query/gallery set
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
        gset = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gset]
        if q:
            dorsal_qg.append((q, g))

    cache_d = {}

    def dist_for(name, q, g, tag):
        key = (name, tag)
        if key not in cache_d:
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            cache_d[key] = champ_dist(q, g, [models[name]], ids, cams_qg)
        return cache_d[key]

    def score_set(sel):
        # mean of per-model champion distances = distance-level ensemble
        dP2 = np.mean([dist_for(n, items, items, "P2") for n in sel], axis=0)
        p2 = _score(items, items, dP2, (1, 5, 10))["rank-1"]
        dP1 = np.mean([dist_for(n, q1, g1, "P1") for n in sel], axis=0)
        p1 = _score(q1, g1, dP1, (1, 5, 10))["rank-1"]
        dors = []
        for i, (q, g) in enumerate(dorsal_qg):
            dd = np.mean([dist_for(n, q, g, f"d{i}") for n in sel], axis=0)
            dors.append(_score(q, g, dd, (1, 5, 10))["rank-1"])
        return p1, float(np.mean(dors)), p2

    # greedy forward selection maximizing P2
    sel = []
    best_p2 = -1
    remaining = list(models)
    print("\ngreedy forward selection (maximize P2):", flush=True)
    while remaining:
        cand = []
        for m in remaining:
            p1, dr, p2 = score_set(sel + [m])
            cand.append((p2, dr, p1, m))
        cand.sort(reverse=True)
        p2, dr, p1, m = cand[0]
        if p2 <= best_p2 + 0.001 and len(sel) >= 3:
            break
        sel.append(m); remaining.remove(m); best_p2 = p2
        print(f"  +{m:9s} -> set({len(sel)})  P1 {p1:.3f} | dorsal {dr:.3f} | "
              f"P2 {p2:.3f}", flush=True)

    p1, dr, p2 = score_set(sel)
    print(f"\nBEST P2 SET ({len(sel)}): {sel}", flush=True)
    print(f"  P1 {p1:.3f} | dorsal {dr:.3f} | P2 {p2:.3f}", flush=True)
    print(f"  all-10 ref: ", end="")
    p1a, dra, p2a = score_set(list(models))
    print(f"P1 {p1a:.3f} | dorsal {dra:.3f} | P2 {p2a:.3f}", flush=True)

    with open("artifacts2/ensemble_search_v1.json", "w", encoding="utf-8") as fh:
        json.dump({"best_set": sel, "best": {"P1": p1, "dorsal": dr, "P2": p2},
                   "all10": {"P1": p1a, "dorsal": dra, "P2": p2a}}, fh, indent=2)
    print("saved artifacts2/ensemble_search_v1.json", flush=True)


if __name__ == "__main__":
    main()
