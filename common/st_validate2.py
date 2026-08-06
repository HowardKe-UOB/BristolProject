"""Pre-screen cheap label-free INFERENCE levers on frozen features (CPU).

Levers, all label-free and training-free:
  * CC   -- per-camera embedding centering (subtract camera mean, renormalise);
            the minimal form of camera-aware distance rectification (UPCA/CMLR
            family). Attacks the oblique-vs-dorsal camera bias directly.
  * AQE  -- alpha-weighted query expansion (standard retrieval trick).
  * ST   -- spatio-temporal impossibility mask (cowreid/st_inference.py).
  * RR   -- CA-Jaccard re-ranking (already in the repo).

    python st_validate2.py --listing 2025Sep18.listing.txt
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import ca_jaccard_distance
from cowreid.eval import EvalItem, _score, _stack
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex


def camera_center(items, X, strength=1.0):
    """Subtract per-camera mean and renormalise (strength in [0,1])."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    by_cam = defaultdict(list)
    for i, it in enumerate(items):
        by_cam[it.camera].append(i)
    out = X.copy()
    for _cam, idx in by_cam.items():
        mu = X[idx].mean(0, keepdims=True)
        out[idx] = X[idx] - strength * mu
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


def aqe(Q, G, dist, k=10, alpha=3.0):
    """Alpha query expansion: expand each query with its top-k gallery neighbours
    (weights = similarity**alpha), using the given (possibly masked) distances."""
    newQ = Q.copy()
    for i in range(Q.shape[0]):
        nn = np.argsort(dist[i], kind="stable")[:k]
        sims = np.clip(1.0 - dist[i][nn], 0.0, None) ** alpha
        newQ[i] = Q[i] + (sims[:, None] * G[nn]).sum(0)
    return newQ / (np.linalg.norm(newQ, axis=1, keepdims=True) + 1e-12)


def run_all(query, gallery, emb, index, topo, margin=0, tag="", report=None):
    q, g = list(query), list(gallery)
    Qr, Gr = _stack(q, emb), _stack(g, emb)
    mask = build_st_mask(q, g, index, topo, margin=margin)
    X = np.concatenate([Qr, Gr], 0)
    cc = camera_center(q + g, X)
    variants = {
        "cosine": (Qr / np.linalg.norm(Qr, axis=1, keepdims=True),
                   Gr / np.linalg.norm(Gr, axis=1, keepdims=True)),
        "CC": (cc[: len(q)], cc[len(q):]),
    }
    results = {}
    for name, (Q, G) in variants.items():
        dist = 1.0 - Q @ G.T
        results[name] = _score(q, g, dist, (1, 5, 10))
        dm = dist.copy(); dm[mask] = INF
        results[name + "+ST"] = _score(q, g, dm, (1, 5, 10))
        Qe = aqe(Q, G, dm)
        de = 1.0 - Qe @ G.T; de[mask] = INF
        results[name + "+ST+AQE"] = _score(q, g, de, (1, 5, 10))
        # re-rank on top of the (possibly centered) embeddings, then mask
        feats = np.concatenate([Q, G], 0)
        cams = [it.camera for it in q] + [it.camera for it in g]
        D = ca_jaccard_distance(feats, cams, k1=20, k2=6, camera_aware=True)
        dr = D[: len(q), len(q):].copy()
        results[name + "+RR"] = _score(q, g, dr, (1, 5, 10))
        dr[mask] = INF
        results[name + "+RR+ST"] = _score(q, g, dr, (1, 5, 10))
    for name, r in results.items():
        print(f"  {tag}{name:14s}: mAP={r['mAP']:.3f} r1={r['rank-1']:.3f} "
              f"r5={r['rank-5']:.3f} r10={r['rank-10']:.3f}", flush=True)
    if report is not None:
        report[tag or "run"] = results
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--out", default="artifacts2/st_inference_frozen_v1.json")
    args = ap.parse_args()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    hold = args.holdout_camera
    gal_ids = {t.gt_label for t in tracklets if t.camera != hold}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != hold]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gal_ids]

    d = np.load(args.cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    emb = {t: fc[t].mean(0) for t in fc}

    print(f"frozen features, leave-out {hold}, |Q|={len(query)} |G|={len(gallery)}", flush=True)
    report = {}
    run_all(query, gallery, emb, index, topo, margin=0, report=report)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
