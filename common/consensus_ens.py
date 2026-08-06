"""Push the LABEL-FREE ensemble toward 0.80 via (1) consensus weighting and
(2) an ensemble-space mining-precision diagnostic. CPU-only, saved embeddings.

(1) CONSENSUS WEIGHTING. The collapsed seed cannot be identified with GT, but it
CAN be identified by disagreement with the other seeds: compute the correlation
of each seed's champion distance matrix with the leave-one-out consensus mean,
then weight seeds by (positive) agreement -- pure majority voting, no labels.
Also test hard-drop of the least-agreeing seed.

(2) MINING DIAGNOSTIC. The gallery embeddings in `_vitb_cap_ens5_emb_v1.npz` ARE
all train-camera tracklets. Measure cross-camera mutual-kNN link precision in
(a) each single seed's space and (b) the ensemble average space. GT is used to
MEASURE precision only. If ensemble-space precision >> 15%, an ensemble-distilled
retrain is justified.

    python consensus_ens.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

HOLD = "66.130"
RANKS = (1, 5, 10)


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:34s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}


def mutual_knn_links(X, cams, k=1):
    """Cross-camera mutual top-k pairs from an (N, D) L2 feature matrix."""
    S = X @ X.T
    N = len(X)
    for i in range(N):
        S[i, i] = -2
        S[i, np.array(cams) == cams[i]] = -2          # cross-camera only
    best = np.argsort(-S, axis=1)[:, :k]
    links = set()
    for i in range(N):
        for j in best[i]:
            if i in best[j]:
                links.add(frozenset((i, j)))
    return links


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
    cams_qg = [it.camera for it in q] + [it.camera for it in g]
    mask = build_st_mask(q, g, index, topo, margin=0)

    d = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    seeds = [k for k in d.files if k != "ids"]

    def champ_dist(M):
        E = {t: M[pos[t]] for t in ids}
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    print("computing per-seed champion distances...", flush=True)
    dists = {s: champ_dist(d[s]) for s in seeds}
    report = {}

    # ---------------- (1) consensus weighting ---------------- #
    print("\n[1] seed-consensus analysis (label-free)", flush=True)
    flat = {s: dists[s].ravel() for s in seeds}
    agree = {}
    for s in seeds:
        loo = np.mean([flat[t] for t in seeds if t != s], axis=0)
        agree[s] = float(np.corrcoef(flat[s], loo)[0, 1])
        print(f"    {s}: corr with leave-one-out consensus = {agree[s]:.4f}", flush=True)
    report["consensus_corr"] = agree
    worst = min(agree, key=agree.get)
    print(f"    least-agreeing seed (label-free pick): {worst}", flush=True)

    print("\n[1b] ensembles:", flush=True)
    show("mean(all 5)", q, g, np.mean(list(dists.values()), axis=0), mask, report)
    w = np.array([max(agree[s], 0.0) for s in seeds]); w = w / w.sum()
    show("consensus-weighted mean", q, g,
         np.sum([wi * dists[s] for wi, s in zip(w, seeds)], axis=0), mask, report)
    w2 = np.array([max(agree[s] - min(agree.values()), 0.0) for s in seeds]); w2 = w2 / w2.sum()
    show("consensus-weighted (sharp)", q, g,
         np.sum([wi * dists[s] for wi, s in zip(w2, seeds)], axis=0), mask, report)
    show(f"mean(drop {worst})", q, g,
         np.mean([dists[s] for s in seeds if s != worst], axis=0), mask, report)

    # ---------------- (2) mining diagnostic ---------------- #
    print("\n[2] cross-camera mutual-NN link precision on TRAIN cameras "
          "(GT for measurement only)", flush=True)
    g_tids = [it.tracklet_id for it in g]
    g_cams = [it.camera for it in g]
    gt_of = {it.tracklet_id: it.identity for it in g}
    Xens = np.mean([d[s] for s in seeds], axis=0)
    Xens = Xens / (np.linalg.norm(Xens, axis=1, keepdims=True) + 1e-12)
    spaces = {s: d[s] for s in seeds}
    spaces["ENSEMBLE-mean"] = Xens
    diag = {}
    for name, M in spaces.items():
        X = np.stack([M[pos[t]] for t in g_tids])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        for k in (1, 2):
            links = mutual_knn_links(X.astype(np.float32), g_cams, k=k)
            correct = sum(gt_of[g_tids[a]] == gt_of[g_tids[b]]
                          for a, b in (tuple(l) for l in links))
            prec = correct / max(len(links), 1)
            diag[f"{name}_k{k}"] = {"links": len(links), "precision": round(prec, 4)}
            print(f"    {name:14s} k={k}: {len(links):4d} links  precision={prec:.3f}", flush=True)
    report["mining_diagnostic"] = diag

    with open("artifacts2/consensus_ens_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nrefs: 5-seed mean 0.749/0.932/0.491 | oracle(no s1) 0.785 | goal 0.80")
    print("saved artifacts2/consensus_ens_v1.json")


if __name__ == "__main__":
    main()
