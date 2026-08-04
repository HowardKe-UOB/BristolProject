"""Diagnose WHY non-oblique query cameras score low (CPU, saved embeddings).

For each dorsal query camera X, evaluate the k=2 trio (holdout mode) under:
  (a) full gallery (all other cameras incl. the oblique 66.130) -- reproduces the
      published sweep numbers;
  (b) dorsal-only gallery (66.130 excluded).
Also break down, for each rank-1 ERROR in (a), whether the wrong top-1 item is an
oblique-camera tracklet, and how many queries have their ONLY true matches in the
oblique camera (unanswerable in (b), skipped by the junk rule).

    python diag_percam_errors.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

OBL = "66.130"
K2 = ("s7", "s8", "s9")


def champ_dist(q, g, embs_list, ids, cams_qg):
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
    by_tid = {t.tracklet_id: t for t in tracklets}

    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"])
    embs = [d[s] for s in sorted(d.files) if any(k in s for k in K2)]

    cams_all = sorted({by_tid[t].camera for t in ids})
    dorsal = [c for c in cams_all if c != OBL]

    print(f"{'query':8s} {'full-gal r1':>11s} {'dorsal-gal r1':>13s} "
          f"{'err@obl':>8s} {'only-obl-match':>14s}", flush=True)
    rows = []
    per_cam = {}
    for X in dorsal:
        # (a) full gallery
        g_full = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                  for t in ids if by_tid[t].camera != X]
        gset = {it.identity for it in g_full}
        q_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera == X and by_tid[t].gt_label in gset]
        cams_qg = [it.camera for it in q_items] + [it.camera for it in g_full]
        dist = champ_dist(q_items, g_full, embs, ids, cams_qg)
        r_full = _score(q_items, g_full, dist, (1, 5, 10))

        # error attribution: for each query, is the top non-junk item oblique?
        g_ids = np.array([it.identity for it in g_full])
        g_cams = np.array([it.camera for it in g_full])
        n_err = n_err_obl = n_only_obl = 0
        for i, qi in enumerate(q_items):
            order = np.argsort(dist[i], kind="stable")
            keep = ~((g_ids[order] == qi.identity) & (g_cams[order] == qi.camera))
            ids_o = g_ids[order][keep]; cams_o = g_cams[order][keep]
            true_cams = set(g_cams[g_ids == qi.identity])
            if true_cams == {OBL}:
                n_only_obl += 1
            if ids_o[0] != qi.identity:
                n_err += 1
                if cams_o[0] == OBL:
                    n_err_obl += 1

        # (b) dorsal-only gallery
        g_dor = [it for it in g_full if it.camera != OBL]
        gset2 = {it.identity for it in g_dor}
        q2 = [it for it in q_items if it.identity in gset2]
        cams_qg2 = [it.camera for it in q2] + [it.camera for it in g_dor]
        dist2 = champ_dist(q2, g_dor, embs, ids, cams_qg2)
        r_dor = _score(q2, g_dor, dist2, (1, 5, 10))

        frac_err_obl = n_err_obl / max(n_err, 1)
        print(f"{X:8s} {r_full['rank-1']:11.3f} {r_dor['rank-1']:13.3f} "
              f"{frac_err_obl:8.1%} {n_only_obl:14d}", flush=True)
        rows.append((X, r_full["rank-1"], r_dor["rank-1"], frac_err_obl, n_only_obl,
                     r_full["mAP"], r_dor["mAP"]))
        per_cam[X] = {
            "full_gallery": r_full,
            "dorsal_only_gallery": r_dor,
            "n_query_full": len(q_items),
            "n_query_dorsal": len(q2),
            "n_rank1_errors": int(n_err),
            "n_rank1_errors_top1_oblique": int(n_err_obl),
            "frac_rank1_errors_top1_oblique": round(float(frac_err_obl), 4),
            "n_query_only_oblique_true_match": int(n_only_obl),
        }

    m_full = np.mean([r[1] for r in rows]); m_dor = np.mean([r[2] for r in rows])
    print(f"\nmean over dorsal queries: full-gallery r1={m_full:.3f}  "
          f"dorsal-only r1={m_dor:.3f}", flush=True)
    print(f"mean mAP: full={np.mean([r[5] for r in rows]):.3f}  "
          f"dorsal-only={np.mean([r[6] for r in rows]):.3f}", flush=True)

    # serialize all computed metrics (no computation change) to a versioned JSON
    out = {
        "script": "diag_percam_errors.py",
        "embeddings_file": "_vitb_dst_emb_v4.npz",
        "seeds": list(K2),
        "oblique_camera": OBL,
        "per_camera": per_cam,
        "mean_over_dorsal_queries": {
            "full_gallery_rank1": round(float(m_full), 4),
            "dorsal_only_rank1": round(float(m_dor), 4),
            "full_gallery_mAP": round(float(np.mean([r[5] for r in rows])), 4),
            "dorsal_only_mAP": round(float(np.mean([r[6] for r in rows])), 4),
        },
    }
    base = os.path.join("artifacts2", "diag_percam_errors")
    v = 1
    while os.path.exists(f"{base}_v{v}.json"):
        v += 1
    path = f"{base}_v{v}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
