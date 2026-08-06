"""QE-hub v2: oblique-camera HUB RELAY reranking on the STRONG heterogeneous
ensemble (zero training, CPU). Dorsal x vs dorsal y is hard (identical backs), but
oblique<->dorsal is reliable (P1 0.926). So route dorsal-dorsal comparisons THROUGH
the oblique camera: two-hop similarity = how much x and y both match the same
oblique tracklets. Gated: only applied to dorsal queries whose top oblique match is
confident (coverage was cowork's failure mode on weak DINOv2; the ensemble's
stronger oblique bridge should fix it).

Uses the current best clean ensemble embeddings; oblique = camera 66.130 gallery
tracklets act as hubs. Distances fused by RRF with the direct champion distance.

    python qe_hub2.py
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
# clean 9-model ensemble (the defensible max-dorsal/P2 combo):
# hc16/17/18 (idx 3,4,5 of final_zerohuman) + mega2 trio + mega2ft trio
SRC = [
    ("_sweep_final_zerohuman_emb.npz", [3, 4, 5]),
    ("_sweep_mega2_trio_emb.npz", [0, 1, 2]),
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2]),
]


def rank_of(dist):
    order = np.argsort(dist, axis=1, kind="stable")
    r = np.empty_like(order)
    for i in range(order.shape[0]):
        r[i, order[i]] = np.arange(order.shape[1])
    return r


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ids = None; embs = []
    for npz, idxs in SRC:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        keys = [k for k in d.files if k != "ids"]
        for i in idxs:
            embs.append(d[keys[i]])
    ipos = {t: i for i, t in enumerate(ids)}
    print(f"{len(embs)} models", flush=True)

    # fused cosine SIMILARITY over all tracklets (dimension-agnostic hub space)
    Sfull = np.zeros((len(ids), len(ids)))
    for M in embs:
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        Sfull += Mn @ Mn.T
    Sfull /= len(embs)

    def champ(q, g):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        return champ_dist(q, g, embs, ids, cams_qg)

    def hub_relay(q, g, hub_ids, tau):
        """two-hop distance via oblique hubs, gated by hub-match confidence."""
        qi = [ipos[it.tracklet_id] for it in q]
        gi = [ipos[it.tracklet_id] for it in g]
        hi = [ipos[h] for h in hub_ids]
        Sqh = Sfull[np.ix_(qi, hi)]                      # (Nq, H) query->hub sim
        Sgh = Sfull[np.ix_(gi, hi)]                      # (Ng, H)
        # soft hub assignment weights (only reliable oblique matches count)
        Wq = np.exp((Sqh - 1.0) / 0.1); Wq /= Wq.sum(1, keepdims=True) + 1e-9
        Wg = np.exp((Sgh - 1.0) / 0.1); Wg /= Wg.sum(1, keepdims=True) + 1e-9
        two_hop_sim = Wq @ Wg.T                           # (Nq, Ng) shared-hub mass
        conf = Sqh.max(1)                                 # per-query hub confidence
        gate = conf >= tau
        return 1.0 - two_hop_sim, gate

    report = {}

    def run(q, g, name, hub_ids):
        base = champ(q, g)
        rb = _score(q, g, base, (1, 5, 10))
        out = {"base": rb["rank-1"]}
        if hub_ids:
            for tau in (0.4, 0.5, 0.6):
                hub_d, gate = hub_relay(q, g, hub_ids, tau)
                # RRF fuse only for gated queries; others keep base
                rk_b = rank_of(base); rk_h = rank_of(hub_d)
                fused = base.copy().astype(float)
                fscore = -(1.0 / (60 + rk_b) + 1.0 / (60 + rk_h))
                for i in range(len(q)):
                    if gate[i]:
                        fused[i] = fscore[i]
                r = _score(q, g, fused, (1, 5, 10))
                out[f"relay@{tau}"] = r["rank-1"]
                out[f"gatefrac@{tau}"] = float(gate.mean())
        report[name] = out
        line = f"  {name:8s} base {out['base']:.3f}"
        if hub_ids:
            for tau in (0.4, 0.5, 0.6):
                line += f" | relay@{tau} {out[f'relay@{tau}']:.3f} ({out[f'gatefrac@{tau}']:.0%})"
        print(line, flush=True)

    # per dorsal camera: hubs = oblique (66.130) tracklets present in gallery
    dorsal_r = defaultdict(list)
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gs = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gs]
        hub_ids = [t for t in ids if by_tid[t].camera == OBL]
        if q:
            run(q, g, f"q_{X}", hub_ids)
            for kk in report[f"q_{X}"]:
                if kk.startswith("relay@") or kk == "base":
                    dorsal_r[kk].append(report[f"q_{X}"][kk])
    print("  dorsal  " + " | ".join(
        f"{k} {np.mean(v):.3f}" for k, v in dorsal_r.items() if not k.startswith("gate")),
        flush=True)

    with open("artifacts2/qe_hub2_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/qe_hub2_v1.json  (dorsal base ref 0.656)", flush=True)


if __name__ == "__main__":
    main()
