"""SUPERVISED-REGIME fusion (USES GT LABELS — NOT zero-human): fuse the supervised
twin with the zero-human heterogeneous ensemble. Measures (a) the ensemble ceiling,
(b) whether the label-free ensemble's strong oblique (P1 0.926 > supervised 0.896)
lifts the supervised model. CPU, saved embeddings.

    python fuse_supervised.py
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
SRC = [
    ("_sweep_final_zerohuman_emb.npz", [3, 4, 5]),      # hc16/17/18
    ("_sweep_mega2_trio_emb.npz", [0, 1, 2]),
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2]),
    ("_sweep_sup_full_emb.npz", [0]),                   # supervised twin (GT!)
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ids = None; models = {}
    for npz, idxs in SRC:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        keys = [k for k in d.files if k != "ids"]
        for i in idxs:
            nm = "SUP" if "sup" in npz else keys[i]
            models[nm] = d[keys[i]]
    zh = [k for k in models if k != "SUP"]

    def score(names, weights=None):
        embs = [models[n] for n in names]
        w = weights or [1.0] * len(names)

        def dist(q, g):
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            ds = [champ_dist(q, g, [e], ids, cams_qg) for e in embs]
            return np.average(ds, axis=0, weights=w)
        g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
        q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
              if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
        p1 = _score(q1, g1, dist(q1, g1), (1, 5, 10))
        dm = []
        for X in sorted({by_tid[t].camera for t in ids}):
            if X == OBL:
                continue
            g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
            gs = {it.identity for it in g}
            q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
                 if by_tid[t].camera == X and gt[t] in gs]
            if q:
                dm.append(_score(q, g, dist(q, g), (1, 5, 10))["rank-1"])
        id_cams = defaultdict(set)
        for t in ids:
            id_cams[gt[t]].add(by_tid[t].camera)
        multi = {i for i, c in id_cams.items() if len(c) >= 2}
        items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
        p2 = _score(items, items, dist(items, items), (1, 5, 10))
        return p1, float(np.mean(dm)), p2

    report = {}
    combos = {
        "SUP only (labels)": (["SUP"], None),
        "zero-human 9 (no labels)": (zh, None),
        "SUP + zero-human 9": (["SUP"] + zh, None),
        "SUP x3 + zero-human 9": (["SUP"] + zh, [3] + [1] * len(zh)),
        "SUP x6 + zero-human 9": (["SUP"] + zh, [6] + [1] * len(zh)),
    }
    print("(SUP = supervised twin, uses GT labels — NOT zero-human)\n", flush=True)
    for name, (names, w) in combos.items():
        p1, dr, p2 = score(names, w)
        print(f"  {name:26s}: P1 {p1['rank-1']:.3f} | dorsal {dr:.3f} | "
              f"P2 {p2['rank-1']:.3f} (r5 {p2['rank-5']:.3f} mAP {p2['mAP']:.3f})",
              flush=True)
        report[name] = {"P1": p1, "dorsal": dr, "P2": p2}
    with open("artifacts2/fuse_supervised_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/fuse_supervised_v1.json", flush=True)


if __name__ == "__main__":
    main()
