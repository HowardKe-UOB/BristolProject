"""Oblique-HUB RELAY for dorsal-dorsal retrieval (CPU only, saved embeddings).

Motivation (PAPER_draft_v1 S5.6 gallery-ablation): removing the oblique camera
66.130 from the gallery makes one dorsal query camera DROP 12 points -- its
easiest true matches lived in the oblique view. So the oblique view already
acts as an *implicit* bridge between dorsal cameras. This script makes the
bridge EXPLICIT at retrieval time, with zero training:

    two-hop similarity   hub(i, j) = max_h min( sim(q_i, h), sim(g_j, h) )

over the oblique-camera gallery tracklets h -- "query and gallery item both
strongly match the SAME oblique view". Oblique<->dorsal matching is the
reliable direction (rank-1 0.88), so agreement through a shared oblique hub is
strong evidence of identity, exactly where dorsal-dorsal appearance fails.
Unlike AQE (which failed at -8.0 by expanding into look-alike dorsal
neighbours), expansion here is restricted to the aligned oblique view.

Variants, per dorsal query camera (gallery = all other cameras, incl. 66.130):
  base        champion trio distance (reproduces the published sweep, ~0.511);
  hub-only    ranking by two-hop distance alone (diagnostic: signal strength);
  +RRF        rrf(base, hub) rank fusion for all queries;
  +RRF gated  fusion only for queries confidently covered by the oblique view
              (max_h sim(q, h) >= tau); tau in {0.4, 0.5, 0.6};
  +QE (RRF)   soft variant: expand q and g into their top-3 oblique hubs
              (similarity-weighted mean), cosine of expansions, rrf with base.
Also evaluated: full transductive P2 with the same fusions.
P1 (66.130 as query) is untouched by design -- its gallery has no oblique hubs.

Label-free: hubs are unlabeled gallery members; ground truth enters scoring
only. Reads existing files only; writes artifacts2/hub_relay_v1.json.
Runtime: a few minutes, CPU. Safe to run alongside GPU training.

    python hub_relay.py
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
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

OBL = "66.130"
K2 = ("s7", "s8", "s9")
TAUS = (0.4, 0.5, 0.6)
QE_K = 3


def norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cc_feats(q, g, M, ids):
    """Camera-centered, L2-normalized features for one model (champion space)."""
    E = {t: M[i] for i, t in enumerate(ids)}
    Qf = np.stack([E[it.tracklet_id] for it in q])
    Gf = np.stack([E[it.tracklet_id] for it in g])
    X = np.concatenate([Qf, Gf])
    cc = camera_center(list(q) + list(g), X)
    return norm(cc[: len(q)]), norm(cc[len(q):])


def champ_dist(q, g, embs_list, ids, cams_qg):
    """Mean of per-model champion-recipe distances (identical to the sweep)."""
    dists = []
    for M in embs_list:
        E = {t: M[i] for i, t in enumerate(ids)}
        Qf = np.stack([E[it.tracklet_id] for it in q])
        Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(list(q) + list(g), X)
        Qcc, Gcc = cc[: len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        dists.append(rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                          dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20))
    return np.mean(dists, axis=0)


def hub_sims(q, g, embs_list, ids):
    """Model-averaged cosine similarity of queries / gallery to oblique hubs."""
    hub_idx = np.array([j for j, it in enumerate(g) if it.camera == OBL])
    Sq = Sg = None
    for M in embs_list:
        Qcc, Gcc = cc_feats(q, g, M, ids)
        H = Gcc[hub_idx]
        sq, sg = Qcc @ H.T, Gcc @ H.T
        Sq = sq if Sq is None else Sq + sq
        Sg = sg if Sg is None else Sg + sg
    return Sq / len(embs_list), Sg / len(embs_list), hub_idx


def maxmin_two_hop(Sq, Sg):
    """hub(i,j) = max_h min(Sq[i,h], Sg[j,h]); loop over hubs, memory-safe."""
    out = np.full((Sq.shape[0], Sg.shape[0]), -1.0, dtype=np.float32)
    for h in range(Sq.shape[1]):
        np.maximum(out, np.minimum.outer(Sq[:, h], Sg[:, h]), out=out)
    return out


def qe_dist(q, g, embs_list, ids, k=QE_K):
    """Expand q and g into their top-k oblique hubs; cosine of expansions."""
    D = None
    for M in embs_list:
        Qcc, Gcc = cc_feats(q, g, M, ids)
        hub_idx = np.array([j for j, it in enumerate(g) if it.camera == OBL])
        H = Gcc[hub_idx]

        def expand(F):
            S = F @ H.T
            top = np.argsort(-S, axis=1)[:, :k]
            W = np.take_along_axis(S, top, axis=1).clip(min=0.0)
            return norm(np.einsum("ik,ikd->id", W, H[top]))

        d = 1.0 - expand(Qcc) @ expand(Gcc).T
        D = d if D is None else D + d
    return D / len(embs_list)


def gate_rows(D_base, D_fused, q_cov, tau):
    """Per-query gate: fused ranking only where the query is hub-covered."""
    out = D_base.copy()
    out[q_cov >= tau] = D_fused[q_cov >= tau]
    return out


def pack(r):
    return {k: r[k] for k in ("rank-1", "rank-5", "mAP")}


def eval_all(q_items, g_items, embs, ids, cams_qg):
    D_base = champ_dist(q_items, g_items, embs, ids, cams_qg)
    Sq, Sg, hub_idx = hub_sims(q_items, g_items, embs, ids)
    D_hub = 1.0 - maxmin_two_hop(Sq, Sg)
    D_rrf = rrf([D_base, D_hub], k=20)
    D_qe = rrf([D_base, qe_dist(q_items, g_items, embs, ids)], k=20)
    q_cov = Sq.max(axis=1)

    out = {"n_query": len(q_items), "n_hubs": int(len(hub_idx)),
           "hub_cov_median": float(np.median(q_cov)),
           "base": pack(_score(q_items, g_items, D_base, (1, 5, 10))),
           "hub_only": pack(_score(q_items, g_items, D_hub, (1, 5, 10))),
           "rrf": pack(_score(q_items, g_items, D_rrf, (1, 5, 10))),
           "qe_rrf": pack(_score(q_items, g_items, D_qe, (1, 5, 10)))}
    for tau in TAUS:
        r = _score(q_items, g_items, gate_rows(D_base, D_rrf, q_cov, tau), (1, 5, 10))
        out[f"rrf_gated@{tau}"] = pack(r)
        out[f"gated_frac@{tau}"] = float((q_cov >= tau).mean())
    return out


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}

    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"])
    embs = [d[s] for s in sorted(d.files) if any(k in s for k in K2)]
    print(f"{len(embs)} k=2 student models, {len(ids)} embedded tracklets", flush=True)

    report = {"params": {"models": K2, "taus": TAUS, "qe_k": QE_K}}
    variants = ["base", "hub_only", "rrf"] + [f"rrf_gated@{t}" for t in TAUS] + ["qe_rrf"]

    # ---------------- per-camera sweep, dorsal queries only ---------------- #
    cams_all = sorted({by_tid[t].camera for t in ids})
    dorsal = [c for c in cams_all if c != OBL]
    print(f"\n[per-camera] dorsal queries, gallery = all other cams (incl. {OBL})",
          flush=True)
    means = defaultdict(list)
    for X in dorsal:
        g_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera != X]
        gset = {it.identity for it in g_items}
        q_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera == X
                   and by_tid[t].gt_label in gset]
        if not q_items:
            continue
        cams_qg = [it.camera for it in q_items] + [it.camera for it in g_items]
        res = eval_all(q_items, g_items, embs, ids, cams_qg)
        report[f"query_{X}"] = res
        for v in variants:
            means[v].append(res[v]["rank-1"])
        print(f"  query={X:8s} base={res['base']['rank-1']:.3f}  "
              f"hub-only={res['hub_only']['rank-1']:.3f}  "
              f"rrf={res['rrf']['rank-1']:.3f}  "
              f"gated@0.5={res['rrf_gated@0.5']['rank-1']:.3f}  "
              f"qe={res['qe_rrf']['rank-1']:.3f}  "
              f"(hubs={res['n_hubs']}, cov_med={res['hub_cov_median']:.2f})",
              flush=True)

    print("\n[dorsal-query mean rank-1]", flush=True)
    for v in variants:
        report[f"dorsal_mean_{v}"] = float(np.mean(means[v]))
        print(f"  {v:14s} {np.mean(means[v]):.3f}", flush=True)

    # ---------------- P2 full transductive ---------------- #
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
             for t in ids if by_tid[t].gt_label in multi]
    cams_qg = [it.camera for it in items] * 2
    print(f"\n[P2] full transductive: {len(items)} items", flush=True)
    res = eval_all(items, items, embs, ids, cams_qg)
    report["P2"] = res
    for v in variants:
        print(f"  {v:14s} r1={res[v]['rank-1']:.3f} r5={res[v]['rank-5']:.3f} "
              f"mAP={res[v]['mAP']:.3f}", flush=True)

    with open("artifacts2/hub_relay_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nsaved artifacts2/hub_relay_v1.json", flush=True)
    print("refs: holdout trio dorsal-query mean 0.511, P2 0.516 (validate_protocols);"
          " deploy trio P2 0.549.", flush=True)
    print("read: if rrf/gated beats base on dorsal mean by >=0.02 with P2 not "
          "worse, the oblique-hub bridge is real -> add to inference stack and "
          "re-run on the hcl/deploy students.", flush=True)


if __name__ == "__main__":
    main()
