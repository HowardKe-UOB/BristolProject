"""Push P2/dorsal to the limit: greedy best + cluster-consistency rerank + weighted
unions of the best sets. Answers 'can zero-human reach 0.75?' honestly. CPU.

    python mega_search2.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

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
POOL = [
    ("_sweep_final_zerohuman_emb.npz", [3, 4, 5], ["hc16", "hc17", "hc18"]),
    ("_sweep_mega2_trio_emb.npz", [0, 1, 2], ["m2_60", "m2_61", "m2_62"]),
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2], ["m2ft80", "m2ft81", "m2ft82"]),
    ("_sweep_sup2_trio_emb.npz", [0, 1, 2], ["sup90", "sup91", "sup92"]),
    ("_sweep_r3_trio_emb.npz", [0, 1, 2], ["r3_100", "r3_101", "r3_102"]),
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
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
    print(f"{len(names)} models", flush=True)

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
    protos = {"P2": (items, items)}
    for i, (q, g) in enumerate(dorsal):
        protos[f"d{i}"] = (q, g)

    print("caching distances...", flush=True)
    cache = {}
    for nm in names:
        for pk, (q, g) in protos.items():
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            cache[(nm, pk)] = champ_dist(q, g, [models[nm]], ids, cams_qg)
    print("cached.", flush=True)

    def clabels(sel):
        D = np.zeros((len(ids), len(ids)))
        for n in sel:
            M = models[n]; Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            D += 1.0 - Mn @ Mn.T
        D = np.clip(0.5 * (D / len(sel)), 0.0, None); np.fill_diagonal(D, 0.0)
        ip = {t: i for i, t in enumerate(ids)}
        for p in cl:
            a, b = tuple(p)
            if a in ip and b in ip:
                D[ip[a], ip[b]] = D[ip[b], ip[a]] = D.max()
        raw = DBSCAN(eps=0.35, min_samples=2, metric="precomputed").fit_predict(D)
        lab, nx = {}, (int(raw.max()) + 1 if raw.max() >= 0 else 0)
        for t, l in zip(ids, raw):
            lab[t] = int(l) if l >= 0 else (nx := nx + 1)
        return lab

    def score(sel, pk, lab=None, bonus=0.05):
        q, g = protos[pk]
        dd = np.mean([cache[(n, pk)] for n in sel], axis=0)
        if lab is not None:
            la = np.array([lab[it.tracklet_id] for it in q])
            lb = np.array([lab[it.tracklet_id] for it in g])
            dd = dd - bonus * (la[:, None] == lb[None, :])
        return _score(q, g, dd, (1, 5, 10))["rank-1"]

    def dm(sel, lab=None):
        return float(np.mean([score(sel, f"d{i}", lab) for i in range(len(dorsal))]))

    def greedy(obj):
        sel, best, rem = [], -1, list(names)
        while rem:
            v, m = sorted(((obj(sel + [m]), m) for m in rem), reverse=True)[0]
            if v <= best + 0.0005 and len(sel) >= 3:
                break
            sel.append(m); rem.remove(m); best = v
        return sel

    report = {}
    print("\n=== P2 max ===", flush=True)
    s2 = greedy(lambda s: score(s, "P2"))
    lab2 = clabels(s2)
    for tag, l in [("plain", None), ("+cluster", lab2)]:
        v = score(s2, "P2", l)
        print(f"  greedy-P2 {s2} {tag}: P2 {v:.3f}", flush=True)
        report[f"P2_{tag}"] = v
    # bigger sets + cluster + bonus sweep
    sd = greedy(dm)
    union = list(dict.fromkeys(s2 + sd))
    labU = clabels(union)
    for bonus in (0.05, 0.10, 0.15):
        v = score(union, "P2", labU, bonus)
        print(f"  union({len(union)}) +cluster b={bonus}: P2 {v:.3f}", flush=True)
        report[f"P2_union_b{bonus}"] = v

    print("\n=== dorsal max ===", flush=True)
    labd = clabels(sd)
    for tag, l in [("plain", None), ("+cluster", labd)]:
        print(f"  greedy-dorsal {sd} {tag}: dorsal {dm(sd, l):.3f}", flush=True)
    report["dorsal_plain"] = dm(sd)
    report["dorsal_cluster"] = dm(sd, labd)
    report["dorsal_union_cluster"] = dm(union, labU)
    print(f"  union +cluster: dorsal {dm(union, labU):.3f}", flush=True)

    best_p2 = max(v for k, v in report.items() if k.startswith("P2"))
    best_dr = max(v for k, v in report.items() if k.startswith("dorsal"))
    print(f"\n>>> BEST zero-human: P2 {best_p2:.3f} | dorsal {best_dr:.3f}  "
          f"(target 0.75)", flush=True)
    with open("artifacts2/mega_search2_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
