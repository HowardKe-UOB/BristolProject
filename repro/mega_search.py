"""Comprehensive greedy ensemble search over the FULL zero-human model zoo.
Caches per-model champion distance per protocol (slow, upfront), then greedy
forward-selects the best subset for P1 / dorsal / P2 independently. Finds the true
ceiling of everything trained. CPU.

    python mega_search.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json
from collections import defaultdict

import numpy as np

from cowreid import Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist

OBL = "66.130"
# (npz, [indices], [names]) — distinct zero-human models only
POOL = [
    ("_sweep_final_zerohuman_emb.npz", [0, 1, 2, 3, 4, 5],
     ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18"]),
    ("_sweep_mega_trio_emb.npz", [0, 1, 2], ["mega40", "mega41", "mega42"]),
    ("_sweep_mega2_trio_emb.npz", [0, 1, 2], ["m2_60", "m2_61", "m2_62"]),
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2], ["m2ft80", "m2ft81", "m2ft82"]),
    ("_sweep_sup2_trio_emb.npz", [0, 1, 2], ["sup90", "sup91", "sup92"]),
    ("_sweep_r3_trio_emb.npz", [0, 1, 2], ["r3_100", "r3_101", "r3_102"]),
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ids = None; models = {}
    for npz, idxs, names in POOL:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        keys = [k for k in d.files if k != "ids"]
        for i, nm in zip(idxs, names):
            models[nm] = d[keys[i]]
    names = list(models)
    print(f"{len(names)} models: {names}", flush=True)

    # protocol query/gallery sets
    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
    dorsal = []
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gs = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gs]
        if q:
            dorsal.append((q, g))
    protos = {"P1": (q1, g1)}
    for i, (q, g) in enumerate(dorsal):
        protos[f"d{i}"] = (q, g)
    protos["P2"] = (items, items)

    print("caching per-model distances (slow)...", flush=True)
    cache = {}
    for nm in names:
        for pk, (q, g) in protos.items():
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            cache[(nm, pk)] = champ_dist(q, g, [models[nm]], ids, cams_qg)
        print(f"  {nm} done", flush=True)

    def r1(sel, pk):
        q, g = protos[pk]
        dd = np.mean([cache[(n, pk)] for n in sel], axis=0)
        return _score(q, g, dd, (1, 5, 10))["rank-1"]

    def dorsal_mean(sel):
        return float(np.mean([r1(sel, f"d{i}") for i in range(len(dorsal))]))

    def greedy(objective):
        sel, best, rem = [], -1, list(names)
        while rem:
            cand = sorted(((objective(sel + [m]), m) for m in rem), reverse=True)
            v, m = cand[0]
            if v <= best + 0.0005 and len(sel) >= 3:
                break
            sel.append(m); rem.remove(m); best = v
        return sel, best

    report = {}
    print("\n=== greedy per protocol ===", flush=True)
    for name, obj in [("P1", lambda s: r1(s, "P1")),
                      ("dorsal", dorsal_mean),
                      ("P2", lambda s: r1(s, "P2"))]:
        sel, best = greedy(obj)
        p1v = r1(sel, "P1"); drv = dorsal_mean(sel); p2v = r1(sel, "P2")
        print(f"  max-{name:6s} {sel}\n    -> P1 {p1v:.3f} | dorsal {drv:.3f} | P2 {p2v:.3f}",
              flush=True)
        report[f"max_{name}"] = {"set": sel, "P1": p1v, "dorsal": drv, "P2": p2v}

    with open("artifacts2/mega_search_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/mega_search_v1.json", flush=True)


if __name__ == "__main__":
    main()
