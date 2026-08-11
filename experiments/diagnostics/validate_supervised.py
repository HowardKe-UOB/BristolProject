"""Same protocol sweep for the SUPERVISED ViT-B reference (CPU, saved embeddings).
If supervised also drops sharply on non-66.130 query cameras, the drop is a
property of the protocols (query-camera difficulty), not method overfitting.

Supervised read-out = its own best: feat768 cosine + CA-Jaccard re-rank (k1=20,k2=6).

    python validate_supervised.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from new_levers import dist_cosine, dist_rerank

def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}

    d = np.load("_vitb_sup_emb_v1.npz", allow_pickle=True)
    ids = list(d["ids"]); M = d["feat768"]
    have = set(ids)

    id_cams = defaultdict(set)
    for t in tracklets:
        if t.tracklet_id in have:
            id_cams[t.gt_label].add(t.camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    E = {t: M[i] for i, t in enumerate(ids)}

    def run(q, g, tag, report):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        Qf = np.stack([E[it.tracklet_id] for it in q])
        Gf = np.stack([E[it.tracklet_id] for it in g])
        r_cos = _score(q, g, dist_cosine(Qf, Gf), (1, 5, 10))
        r_rr = _score(q, g, dist_rerank(Qf, Gf, cams_qg, k1=20, k2=6), (1, 5, 10))
        print(f"    {tag:22s}: cosine r1={r_cos['rank-1']:.3f} r5={r_cos['rank-5']:.3f} "
              f"mAP={r_cos['mAP']:.3f}  | +RR r1={r_rr['rank-1']:.3f} "
              f"r5={r_rr['rank-5']:.3f} mAP={r_rr['mAP']:.3f}", flush=True)
        report[tag] = {"cosine": r_cos, "rerank": r_rr}

    report = {}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
             for t in ids if by_tid[t].gt_label in multi]
    print(f"[A] FULL transductive ({len(items)} items), SUPERVISED:", flush=True)
    run(items, items, "full_transductive", report)

    print(f"\n[B] per-camera query sweep, SUPERVISED:", flush=True)
    cams_all = sorted({by_tid[t].camera for t in ids})
    for X in cams_all:
        g_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera != X]
        g_set = {it.identity for it in g_items}
        q_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera == X and by_tid[t].gt_label in g_set]
        if q_items:
            run(q_items, g_items, f"query_{X}", report)

    with open("artifacts2/validate_supervised_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/validate_supervised_v1.json")


if __name__ == "__main__":
    main()
