"""Rigorously confirm the CAP 0.804 result on the SAVED embeddings (CPU only):
reproduce the champion RRF recipe, test its stability to the RRF k parameter and
to each fused component, and compare CAP vs prior champion under identical code.

    python cap_confirm.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

HOLD = "66.130"
R = (1, 5, 10)


def parts(q, g, Qf, Gf, cams):
    X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
    Qcc, Gcc = cc[:len(q)], cc[len(q):]
    Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
    return {"CC": dist_cosine(Qcc, Gcc), "PCAW": dist_cosine(Qw, Gw),
            "cos": dist_cosine(Qf, Gf),
            "CC-RR": dist_rerank(Qcc, Gcc, cams, k1=30, k2=6)}


def sc(q, g, dist, mask):
    r = _score(q, g, dist, R); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, R)
    return r, rs


def line(name, q, g, dist, mask, bucket=None):
    r, rs = sc(q, g, dist, mask)
    print(f"  {name:34s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}")
    if bucket is not None:
        bucket[name] = {"plain": r, "st": rs}


def load(npz, key, q, g):
    d = np.load(npz, allow_pickle=True); idx = {t: i for i, t in enumerate(d["ids"])}
    F = d[key]
    return (np.stack([F[idx[it.tracklet_id]] for it in q]),
            np.stack([F[idx[it.tracklet_id]] for it in g]))


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
    q, g = list(query), list(gallery)
    cams = [it.camera for it in q] + [it.camera for it in g]
    mask = build_st_mask(q, g, index, topo, margin=0)

    report = {}
    for tag, npz, key in [("CHAMPION+TTA", "_vitb_unsup_tta_emb_v1.npz", "feat768"),
                          ("CAP (no TTA)", "_vitb_cap_emb_v1.npz", "feat768"),
                          ("CAP + TTA", "_vitb_cap_emb_v1.npz", "feat768_tta")]:
        Qf, Gf = load(npz, key, q, g)
        P = parts(q, g, Qf, Gf, cams)
        print(f"\n### {tag} ({npz}:{key}) ###")
        sec = report[tag] = {"embeddings_file": npz, "key": key}
        for nm in ("cos", "CC", "PCAW", "CC-RR"):
            line(nm, q, g, P[nm], mask, sec)
        print("  -- RRF(CC,PCAW,CC-RR) k-stability --")
        for k in (10, 20, 40, 60):
            line(f"RRF k={k}", q, g, rrf([P["CC"], P["PCAW"], P["CC-RR"]], k=k), mask, sec)
        print("  -- leave-one-component-out (k=20) --")
        line("RRF(PCAW,CC-RR)", q, g, rrf([P["PCAW"], P["CC-RR"]], k=20), mask, sec)
        line("RRF(CC,CC-RR)", q, g, rrf([P["CC"], P["CC-RR"]], k=20), mask, sec)
        line("RRF(CC,PCAW)", q, g, rrf([P["CC"], P["PCAW"]], k=20), mask, sec)

    import json
    out = "artifacts2/cap_confirm_v1.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"script": "cap_confirm.py", "protocol": "P1 leave-out 66.130",
                   "note": "full stack = RRF k=20; leave-one-out rows drop one component",
                   **report}, fh, indent=1)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
