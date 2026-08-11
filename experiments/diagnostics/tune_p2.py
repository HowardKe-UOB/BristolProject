"""P2-specific inference tuning (CPU, saved 4-model embeddings).

The champion recipe's hyper-parameters (CA-Jaccard k1=30/k2=6, RRF k=20) were
tuned on P1. This sweeps them on P2 with the current best model set
(deploy trio + hardCL-v2 student), plus the spatio-temporal mask, and reports
the P1 side of each variant to guard against trade-offs.

    python tune_p2.py
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
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

OBL = "66.130"


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    by_tid = {t.tracklet_id: t for t in tracklets}

    d = np.load("_sweep_dep3_hc2_emb.npz", allow_pickle=True)
    ids = list(d["ids"])
    embs = [d[k] for k in d.files if k != "ids"]
    print(f"{len(embs)} models, {len(ids)} tracklets", flush=True)

    # P2 items
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
             if by_tid[t].gt_label in multi]
    cams2 = [it.camera for it in items] * 2
    mask2 = build_st_mask(items, items, index, topo, margin=0)

    # P1 items (guard)
    g1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL
          and by_tid[t].gt_label in {x.identity for x in g1}]
    cams1 = [it.camera for it in q1] + [it.camera for it in g1]

    def views(q, g, cams_qg, k1, k2):
        per_model = []
        for M in embs:
            E = {t: M[i] for i, t in enumerate(ids)}
            Qf = np.stack([E[it.tracklet_id] for it in q])
            Gf = np.stack([E[it.tracklet_id] for it in g])
            X = np.concatenate([Qf, Gf]); cc = camera_center(list(q) + list(g), X)
            Qcc, Gcc = cc[:len(q)], cc[len(q):]
            Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
            per_model.append({
                "cos": dist_cosine(Qcc, Gcc),
                "pcaw": dist_cosine(Qw, Gw),
                "rr": dist_rerank(Qcc, Gcc, cams_qg, k1=k1, k2=k2),
            })
        return per_model

    report = {}

    def run(tag, q, g, cams_qg, mask, k1, k2, rrfk, use_st):
        pm = views(q, g, cams_qg, k1, k2)
        dm = np.mean([rrf([v["cos"], v["pcaw"], v["rr"]], k=rrfk) for v in pm], axis=0)
        if use_st:
            dm = dm.copy(); dm[mask] = INF
        return _score(q, g, dm, (1, 5, 10))

    print("\n-- P2 sweep (rank-1 / rank-5 / mAP) --", flush=True)
    best = None
    for k1 in (20, 30, 45):
        for k2 in (6, 10):
            for rrfk in (20, 40):
                for st in (False, True):
                    r = run("P2", items, items, cams2, mask2, k1, k2, rrfk, st)
                    name = f"k1={k1} k2={k2} rrfk={rrfk} st={int(st)}"
                    print(f"  {name:28s}: {r['rank-1']:.3f} / {r['rank-5']:.3f} / "
                          f"{r['mAP']:.3f}", flush=True)
                    report[name] = r
                    if best is None or r["rank-1"] > best[1]["rank-1"]:
                        best = (name, r, (k1, k2, rrfk, st))
    print(f"\nbest P2 config: {best[0]} -> r1={best[1]['rank-1']:.3f}", flush=True)

    k1, k2, rrfk, st = best[2]
    r1side = run("P1", q1, g1, cams1, None, k1, k2, rrfk, False)
    print(f"P1 with best-P2 config (no ST): r1={r1side['rank-1']:.3f} "
          f"r5={r1side['rank-5']:.3f} mAP={r1side['mAP']:.3f}", flush=True)
    report["P1_at_bestP2"] = r1side

    with open("artifacts2/tune_p2_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/tune_p2_v1.json  (ref: default config P2 0.554, "
          "P1 0.896)", flush=True)


if __name__ == "__main__":
    main()
