"""Specialist fusion (CPU): combine the P1-strong baseline student (s8) with the
dorsal-strong hardCL student (s15) from their saved sweep embeddings, and report
all protocols. Tests whether complementary specialists lift dorsal without
sacrificing the oblique benchmark.

    python fuse_specialists.py
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


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return list(d["ids"]), [d[k] for k in d.files if k != "ids"]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}

    ids8, m8 = load("_sweep_base_s8_emb.npz")
    ids15, m15 = load("_sweep_hcl_s15_emb.npz")
    assert ids8 == ids15, "id order mismatch"
    ids = ids8

    dv4 = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids_v4 = list(dv4["ids"])
    assert ids_v4 == ids, "trio npz id order mismatch"
    trio = [dv4[s] for s in sorted(dv4.files)
            if any(k in s for k in ("s7", "s8", "s9"))]

    combos = {
        "s8 only": m8,
        "s15 only": m15,
        "s8 + s15": m8 + m15,
        "trio (baseline)": trio,
        "trio + s15": trio + m15,
        "trio + s15 x2": trio + m15 * 2,
    }
    report = {}
    for name, embs in combos.items():
        # P1
        g1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
              if by_tid[t].camera != OBL]
        q1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
              if by_tid[t].camera == OBL
              and by_tid[t].gt_label in {x.identity for x in g1}]
        cams_qg = [it.camera for it in q1] + [it.camera for it in g1]
        p1 = _score(q1, g1, champ_dist(q1, g1, embs, ids, cams_qg), (1, 5, 10))
        # dorsal sweep
        dorsal = []
        for X in sorted({by_tid[t].camera for t in ids}):
            if X == OBL:
                continue
            g = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
                 if by_tid[t].camera != X]
            gset = {it.identity for it in g}
            q = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
                 if by_tid[t].camera == X and by_tid[t].gt_label in gset]
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            r = _score(q, g, champ_dist(q, g, embs, ids, cams_qg), (1, 5, 10))
            dorsal.append(r["rank-1"])
        # P2
        id_cams = defaultdict(set)
        for t in ids:
            id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
        multi = {i for i, c in id_cams.items() if len(c) >= 2}
        items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
                 if by_tid[t].gt_label in multi]
        cams_qg = [it.camera for it in items] * 2
        p2 = _score(items, items, champ_dist(items, items, embs, ids, cams_qg),
                    (1, 5, 10))
        print(f"{name:10s}: P1 r1={p1['rank-1']:.3f} r5={p1['rank-5']:.3f} | "
              f"dorsal mean={np.mean(dorsal):.3f} | P2 r1={p2['rank-1']:.3f} "
              f"mAP={p2['mAP']:.3f}", flush=True)
        report[name] = {"P1": p1, "dorsal_mean_r1": float(np.mean(dorsal)), "P2": p2}

    with open("artifacts2/fuse_specialists_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/fuse_specialists_v1.json")


if __name__ == "__main__":
    main()
