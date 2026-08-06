"""Quick fusion test: add the strong n_stage=2 Mega student (mega2_s60) to the
existing zero-human combos. CPU, saved embeddings.

    python fuse_mega2.py
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
    ("_sweep_final_zerohuman_emb.npz", ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18"]),
    ("_sweep_mega_trio_emb.npz", ["mega40", "mega41", "mega42"]),
    ("_sweep_megaft_s50_emb.npz", ["megaft50"]),
    ("_sweep_mega2_s60_emb.npz", ["mega2_60"]),
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ids = None; models = {}
    for npz, names in SRC:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        for name, k in zip(names, [k for k in d.files if k != "ids"]):
            models[name] = d[k]

    def evalset(names):
        embs = [models[n] for n in names]

        def dist(q, g):
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            return champ_dist(q, g, embs, ids, cams_qg)
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

    combos = {
        "mega2_60 (single)": ["mega2_60"],
        "dino6 + mega2_60": ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18", "mega2_60"],
        "dino6 + mega3 + mega2_60": ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18",
                                     "mega40", "mega41", "mega42", "mega2_60"],
        "best4 + mega2_60": ["hc16", "mega40", "hc18", "megaft50", "mega2_60"],
        "hc16+hc18+mega2_60+megaft50": ["hc16", "hc18", "mega2_60", "megaft50"],
        "dep+hc + mega2_60 + megaft50": ["dep10", "dep11", "dep12", "hc16", "hc17", "hc18",
                                         "mega2_60", "megaft50"],
    }
    report = {}
    for name, names in combos.items():
        p1, dr, p2 = evalset(names)
        print(f"{name:32s}: P1 {p1['rank-1']:.3f} | dorsal {dr:.3f} | "
              f"P2 {p2['rank-1']:.3f} (r5 {p2['rank-5']:.3f} mAP {p2['mAP']:.3f})",
              flush=True)
        report[name] = {"P1": p1, "dorsal": dr, "P2": p2}
    with open("artifacts2/fuse_mega2_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/fuse_mega2_v1.json", flush=True)


if __name__ == "__main__":
    main()
