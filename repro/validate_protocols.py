"""Protocol-robustness validation of the k=2 ensemble-distilled trio (CPU-only).

Protocols:
  A. FULL TRANSDUCTIVE cross-camera: query = gallery = every embedded tracklet
     whose identity appears in >=2 cameras (Market junk rule forces cross-camera).
     Historical label-free best on this protocol ~ rank-1 0.34.
  B. PER-CAMERA QUERY SWEEP: for each camera X, query = X's tracklets (with a
     cross-camera true match), gallery = all other cameras. X=66.130 reproduces
     the headline 0.883 and is the only camera NEVER SEEN in training (strict
     leave-out); other cameras were seen unlabeled (standard transductive USL).

Caveat noted in output: ~17 identities appear ONLY in 66.130; their tracklets were
never embedded, so galleries that would contain them lack those distractors.

    python validate_protocols.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json
from collections import defaultdict

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

K2 = ("s7", "s8", "s9")


def champ_dist(q, g, embs_list, ids, index, topo, cams_qg):
    """Mean of per-model champion-recipe distances (the k=2 trio ensemble)."""
    dists = []
    for M in embs_list:
        E = {t: M[i] for i, t in enumerate(ids)}
        Qf = np.stack([E[it.tracklet_id] for it in q])
        Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(list(q) + list(g), X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        dists.append(rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                          dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20))
    return np.mean(dists, axis=0)


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"])
    have = set(ids)
    embs = [d[s] for s in sorted(d.files) if any(k in s for k in K2)]
    print(f"{len(embs)} k=2 student models, {len(ids)} embedded tracklets", flush=True)

    by_tid = {t.tracklet_id: t for t in tracklets}
    id_cams = defaultdict(set)
    for t in tracklets:
        if t.tracklet_id in have:
            id_cams[t.gt_label].add(t.camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}

    report = {}

    # ---------------- A. full transductive ---------------- #
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
             for t in ids if by_tid[t].gt_label in multi]
    cams_qg = [it.camera for it in items] + [it.camera for it in items]
    print(f"\n[A] FULL transductive: {len(items)} items (identities in >=2 cams)",
          flush=True)
    dist = champ_dist(items, items, embs, ids, index, topo, cams_qg)
    mask = build_st_mask(items, items, index, topo, margin=0)
    r = _score(items, items, dist, (1, 5, 10))
    dm = dist.copy(); dm[mask] = INF
    rs = _score(items, items, dm, (1, 5, 10))
    print(f"    k2-trio ens : r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
          f"r10={r['rank-10']:.3f} mAP={r['mAP']:.3f}", flush=True)
    print(f"    +ST         : r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} "
          f"r10={rs['rank-10']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report["full_transductive"] = {"plain": r, "st": rs}

    # ---------------- B. per-camera query sweep ---------------- #
    print(f"\n[B] per-camera query sweep (gallery = all other cameras):", flush=True)
    cams_all = sorted({by_tid[t].camera for t in ids})
    for X in cams_all:
        g_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera != X]
        g_ids_set = {it.identity for it in g_items}
        q_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera == X
                   and by_tid[t].gt_label in g_ids_set]
        if not q_items:
            continue
        cams_qg = [it.camera for it in q_items] + [it.camera for it in g_items]
        dist = champ_dist(q_items, g_items, embs, ids, index, topo, cams_qg)
        mask = build_st_mask(q_items, g_items, index, topo, margin=0)
        r = _score(q_items, g_items, dist, (1, 5, 10))
        dm = dist.copy(); dm[mask] = INF
        rs = _score(q_items, g_items, dm, (1, 5, 10))
        seen = "UNSEEN in training" if X == "66.130" else "seen unlabeled"
        print(f"    query={X:7s} |Q|={len(q_items):4d}: r1={r['rank-1']:.3f} "
              f"r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}  |+ST r1={rs['rank-1']:.3f} "
              f"({seen})", flush=True)
        report[f"query_{X}"] = {"plain": r, "st": rs, "n_query": len(q_items)}

    with open("artifacts2/validate_protocols_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nnote: ~17 identities appear only in 66.130 and are not embedded; "
          "galleries lack those distractors.", flush=True)
    print("saved artifacts2/validate_protocols_v1.json")


if __name__ == "__main__":
    main()
