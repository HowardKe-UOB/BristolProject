"""Round-2: combine the round-1 winners (PCA-whitening, per-camera whitening,
tuned CA-Jaccard k1=30/k2=6, reciprocal-rank fusion) into the best label-free
recipe, and check whether the same recipe also lifts the SUPERVISED embeddings.
CPU-only, on saved npz. Compares to unsup best RRF(CC,CC-RR)=0.693.

    python new_levers2.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import (camera_center, dist_cosine, dist_rerank, pca_whiten,
                        per_camera_whiten, rrf)

HOLD = "66.130"
RANKS = (1, 5, 10)


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS)
    dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:30s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}
    return r


def build(q, g, cams, Qf, Gf):
    """Return the family of distance matrices used for fusion."""
    X = np.concatenate([Qf, Gf])
    cc = camera_center(q + g, X)
    Qcc, Gcc = cc[:len(q)], cc[len(q):]
    Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
    cw = per_camera_whiten(q + g, X)
    Qcw, Gcw = cw[:len(q)], cw[len(q):]
    return {
        "cos": dist_cosine(Qf, Gf),
        "CC": dist_cosine(Qcc, Gcc),
        "PCAW": dist_cosine(Qw, Gw),
        "CW": dist_cosine(Qcw, Gcw),
        "CC-RR": dist_rerank(Qcc, Gcc, cams, k1=30, k2=6),
        "CW-RR": dist_rerank(Qcw, Gcw, cams, k1=30, k2=6),
        "PCAW-RR": dist_rerank(Qw, Gw, cams, k1=30, k2=6),
    }


def run(tag, q, g, cams, Qf, Gf, mask, report):
    D = build(q, g, cams, Qf, Gf)
    print(f"\n### {tag} ###", flush=True)
    print("-- singles --")
    for k in ("cos", "CC", "PCAW", "CW", "CC-RR", "CW-RR"):
        show(f"{tag}/{k}", q, g, D[k], mask, report)
    print("-- fusions --")
    show(f"{tag}/RRF(CC,CC-RR)", q, g, rrf([D["CC"], D["CC-RR"]]), mask, report)
    show(f"{tag}/RRF(CC,PCAW,CC-RR)", q, g, rrf([D["CC"], D["PCAW"], D["CC-RR"]]), mask, report)
    show(f"{tag}/RRF(CC,CW,PCAW)", q, g, rrf([D["CC"], D["CW"], D["PCAW"]]), mask, report)
    show(f"{tag}/RRF(CC,PCAW,CC-RR,CW-RR)", q, g,
         rrf([D["CC"], D["PCAW"], D["CC-RR"], D["CW-RR"]]), mask, report)
    for kk in (20, 40, 60):
        show(f"{tag}/RRF(CC,PCAW,CC-RR) k={kk}", q, g,
             rrf([D["CC"], D["PCAW"], D["CC-RR"]], k=kk), mask, report)
    # weighted: emphasise the re-rank view (best rank-1) + PCAW (best rank-5)
    show(f"{tag}/RRF(CC,PCAW,CC-RR,CC-RR)", q, g,
         rrf([D["CC"], D["PCAW"], D["CC-RR"], D["CC-RR"]]), mask, report)


def load(npz, q, g):
    d = np.load(npz, allow_pickle=True)
    idx = {t: i for i, t in enumerate(d["ids"])}
    F = d["feat768"]
    Qf = np.stack([F[idx[it.tracklet_id]] for it in q])
    Gf = np.stack([F[idx[it.tracklet_id]] for it in g])
    return Qf, Gf


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
    Qf, Gf = load("_vitb_unsup_emb_v1.npz", q, g)
    run("UNSUP", q, g, cams, Qf, Gf, mask, report)
    Qs, Gs = load("_vitb_sup_emb_v1.npz", q, g)
    run("SUP", q, g, cams, Qs, Gs, mask, report)

    with open("artifacts2/new_levers2_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nsaved artifacts2/new_levers2_v1.json")


if __name__ == "__main__":
    main()
