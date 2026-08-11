"""New label-free INFERENCE levers for unsupervised cross-camera cattle Re-ID.

All operate on the SAVED ViT-B unsupervised embeddings (`_vitb_unsup_emb_v1.npz`,
768-d pooled backbone feature) -- NO retraining, NO labels, CPU-only. Each idea is
scored on leave-out 66.130 and compared to the current best (feat768 CC+RR,
rank-1 0.663 / mAP 0.408).

Ideas (mine, motivated by the retrieval literature + this dataset's biases):
  W   PCA-whitening       decorrelate feat dims (fit on gallery only, label-free).
                          Cattle-coat features are highly correlated -> cosine is
                          dominated by a few global-illumination directions;
                          whitening equalises them.
  CW  per-camera whitening  camera-conditioned mean+std standardisation (stronger
                          form of CC that also fixes per-camera variance/scale, the
                          oblique-vs-dorsal bias).
  DBA database augmentation replace each gallery vector by a kNN-weighted average
                          (denoises the gallery side; complements query expansion,
                          which hurt here).
  RRF reciprocal-rank fusion  fuse the rankings of cosine + CC + CA-Jaccard rather
                          than trusting one distance; robust to any single view's
                          failure.
  ENS emb256 + feat768 fusion  the projection head and the backbone feature encode
                          partly complementary cues; average their distances.
  KSW k1/k2 re-rank sweep  the CA-Jaccard neighbourhood sizes were never tuned for
                          this small, sparse-overlap gallery.

    python new_levers.py
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
from cowreid.cajaccard import ca_jaccard_distance
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex

HOLD = "66.130"
RANKS = (1, 5, 10)


# --------------------------------------------------------------------------- #
# feature transforms (all label-free)
# --------------------------------------------------------------------------- #
def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def pca_whiten(fit_X, apply_Xs, n_dim=None, eps=1e-3, shrink=0.0):
    """PCA-whitening fit on fit_X (gallery), applied to each of apply_Xs."""
    mu = fit_X.mean(0, keepdims=True)
    Xc = fit_X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if n_dim:
        Vt, S = Vt[:n_dim], S[:n_dim]
    d = S / np.sqrt(len(fit_X))
    W = Vt.T / (d * (1 - shrink) + d.mean() * shrink + eps)         # (D, k)
    return [l2((X - mu) @ W) for X in apply_Xs]


def per_camera_whiten(items, X, strength=1.0):
    """Per-camera z-score: subtract cam-mean and divide by cam-std, then renorm."""
    X = l2(X)
    out = X.copy()
    by_cam = defaultdict(list)
    for i, it in enumerate(items):
        by_cam[it.camera].append(i)
    for _c, idx in by_cam.items():
        mu = X[idx].mean(0, keepdims=True)
        sd = X[idx].std(0, keepdims=True) + 1e-6
        out[idx] = (X[idx] - strength * mu) / (sd ** strength)
    return l2(out)


def camera_center(items, X, strength=1.0):
    X = l2(X)
    out = X.copy()
    by_cam = defaultdict(list)
    for i, it in enumerate(items):
        by_cam[it.camera].append(i)
    for _c, idx in by_cam.items():
        out[idx] = X[idx] - strength * X[idx].mean(0, keepdims=True)
    return l2(out)


def dba(G, k=6, alpha=3.0):
    """Database-side augmentation: each gallery vector += kNN-weighted neighbours."""
    G = l2(G)
    sim = G @ G.T
    np.fill_diagonal(sim, -1)
    out = G.copy()
    for i in range(len(G)):
        nn = np.argsort(-sim[i])[:k]
        w = np.clip(sim[i][nn], 0, None) ** alpha
        out[i] = G[i] + (w[:, None] * G[nn]).sum(0)
    return l2(out)


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def dist_cosine(Q, G):
    return 1.0 - l2(Q) @ l2(G).T


def dist_rerank(Q, G, cams, k1=20, k2=6):
    feats = np.concatenate([Q, G], 0)
    D = ca_jaccard_distance(feats, cams, k1=k1, k2=k2, camera_aware=True)
    return D[: len(Q), len(Q):].copy()


def ranks_from_dist(dist):
    """rank position of each gallery item per query (0 = closest)."""
    order = np.argsort(dist, axis=1, kind="stable")
    r = np.empty_like(order)
    for i in range(order.shape[0]):
        r[i, order[i]] = np.arange(order.shape[1])
    return r


def rrf(dists, k=60):
    """Reciprocal-rank fusion of several distance matrices -> a fused distance."""
    fused = np.zeros_like(dists[0], dtype=np.float64)
    for d in dists:
        fused += 1.0 / (k + ranks_from_dist(d))
    return -fused                                     # higher score -> smaller dist


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS)
    dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:26s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"   |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}
    return r


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

    d = np.load("_vitb_unsup_emb_v1.npz", allow_pickle=True)
    idx = {t: i for i, t in enumerate(d["ids"])}
    F = d["feat768"]; P = d["emb256"]
    Qf = np.stack([F[idx[it.tracklet_id]] for it in q])
    Gf = np.stack([F[idx[it.tracklet_id]] for it in g])
    Qp = np.stack([P[idx[it.tracklet_id]] for it in q])
    Gp = np.stack([P[idx[it.tracklet_id]] for it in g])

    print(f"leave-out {HOLD}  |Q|={len(q)} |G|={len(g)}  (feat768 unsupervised)\n", flush=True)
    report = {}

    print("-- baselines --")
    show("feat768 cosine", q, g, dist_cosine(Qf, Gf), mask, report)
    Qcc, Gcc = camera_center(q + g, np.concatenate([Qf, Gf]))[:len(q)], \
               camera_center(q + g, np.concatenate([Qf, Gf]))[len(q):]
    show("feat768 CC", q, g, dist_cosine(Qcc, Gcc), mask, report)
    show("feat768 CC+RR", q, g, dist_rerank(Qcc, Gcc, cams), mask, report)

    print("\n-- W: PCA-whitening (fit on gallery) --")
    for nd in (128, 256, 512):
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=nd)
        show(f"feat768 PCAW{nd}", q, g, dist_cosine(Qw, Gw), mask, report)
    Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
    show("feat768 PCAW256+RR", q, g, dist_rerank(Qw, Gw, cams), mask, report)

    print("\n-- CW: per-camera whitening --")
    X = np.concatenate([Qf, Gf])
    Xcw = per_camera_whiten(q + g, X)
    show("feat768 CW", q, g, dist_cosine(Xcw[:len(q)], Xcw[len(q):]), mask, report)
    show("feat768 CW+RR", q, g, dist_rerank(Xcw[:len(q)], Xcw[len(q):], cams), mask, report)

    print("\n-- DBA: database-side augmentation (on CC) --")
    Gdba = dba(Gcc, k=6, alpha=3.0)
    show("feat768 CC+DBA", q, g, dist_cosine(Qcc, Gdba), mask, report)
    show("feat768 CC+DBA+RR", q, g, dist_rerank(Qcc, Gdba, cams), mask, report)

    print("\n-- KSW: CA-Jaccard k1/k2 sweep (on CC) --")
    best = None
    for k1 in (8, 12, 20, 30):
        for k2 in (2, 3, 6):
            r = show(f"CC+RR k1={k1} k2={k2}", q, g,
                     dist_rerank(Qcc, Gcc, cams, k1=k1, k2=k2), mask, report)
            if best is None or r["rank-1"] > best[1]:
                best = ((k1, k2), r["rank-1"], r["mAP"])
    print(f"  best k1/k2 by rank-1: {best[0]} -> r1={best[1]:.3f} mAP={best[2]:.3f}")

    print("\n-- RRF: reciprocal-rank fusion (cosine + CC + CC-RR) --")
    dcos = dist_cosine(Qf, Gf)
    dcc = dist_cosine(Qcc, Gcc)
    drr = dist_rerank(Qcc, Gcc, cams)
    show("RRF(cos,CC,CC-RR)", q, g, rrf([dcos, dcc, drr]), mask, report)
    show("RRF(CC,CC-RR)", q, g, rrf([dcc, drr]), mask, report)

    print("\n-- ENS: emb256 + feat768 distance fusion --")
    dp = dist_cosine(Qp, Gp)
    show("ENS cos(feat768+emb256)", q, g, 0.5 * dcos + 0.5 * dp, mask, report)
    Qpc = camera_center(q + g, np.concatenate([Qp, Gp]))[:len(q)]
    Gpc = camera_center(q + g, np.concatenate([Qp, Gp]))[len(q):]
    dpc = dist_cosine(Qpc, Gpc)
    show("RRF(CCf,CCp,CC-RRf)", q, g, rrf([dcc, dpc, drr]), mask, report)

    with open("artifacts2/new_levers_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nsaved artifacts2/new_levers_v1.json")


if __name__ == "__main__":
    main()
