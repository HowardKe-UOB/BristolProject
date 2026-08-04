"""BODY-SIZE CUE diagnostic + fusion (zero-human, CPU + file-header reads).

Ceiling cameras are at fixed heights, so detection-box pixel size correlates with
the animal's physical size -- an identity cue fully independent of coat
appearance, and one this project has NEVER used. The crop files on disk ARE the
raw boxes (e.g. 137x439), so per-tracklet size features are free metadata.

Steps:
  1. per tracklet: median log-area and aspect over its 8 sampled frames,
     z-scored WITHIN camera (removes lens/height differences);
  2. discriminative power: AUC of size-distance separating same-cow vs
     different-cow cross-camera pairs (GT for measurement only);
  3. fusion: add the size distance to the champion ensemble distance
     (several weights) and score P1 / dorsal sweep / P2.

    python sizecue_diag.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist
from train_phase2_run import sample_frames

OBL = "66.130"
WORK = "_crops_train"


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    ds = np.load("_sweep_dep3_hc2_emb.npz", allow_pickle=True)
    sids = list(ds["ids"])
    embs = [ds[k] for k in ds.files if k != "ids"]

    # ---- 1. size features per tracklet ---- #
    feats = {}
    miss = 0
    for t in sids:
        tr = by_tid[t]
        sizes = []
        for p in sample_frames(tr, 8):
            fp = os.path.join(WORK, p)
            if os.path.exists(fp):
                with Image.open(fp) as im:
                    sizes.append(im.size)                # (w, h) header only
        if not sizes:
            miss += 1
            continue
        w = np.median([s[0] for s in sizes]); h = np.median([s[1] for s in sizes])
        feats[t] = (np.log(w * h), h / w)
    print(f"size features for {len(feats)}/{len(sids)} tracklets ({miss} missing)",
          flush=True)

    # per-camera z-score
    by_cam = defaultdict(list)
    for t, (la, asp) in feats.items():
        by_cam[by_tid[t].camera].append(t)
    z = {}
    for c, ts in by_cam.items():
        A = np.array([feats[t][0] for t in ts]); P = np.array([feats[t][1] for t in ts])
        za = (A - A.mean()) / (A.std() + 1e-9)
        zp = (P - P.mean()) / (P.std() + 1e-9)
        for t, a, p in zip(ts, za, zp):
            z[t] = (a, p)

    # ---- 2. AUC same vs different (cross-camera pairs) ---- #
    rng = np.random.default_rng(0)
    ts_all = [t for t in sids if t in z]
    same_d, diff_d = [], []
    for _ in range(200000):
        a, b = ts_all[rng.integers(len(ts_all))], ts_all[rng.integers(len(ts_all))]
        if a == b or by_tid[a].camera == by_tid[b].camera:
            continue
        d = abs(z[a][0] - z[b][0])
        (same_d if gt[a] == gt[b] else diff_d).append(d)
    same_d, diff_d = np.array(same_d), np.array(diff_d)
    auc = float(np.mean(rng.choice(diff_d, 50000) > rng.choice(same_d, 50000)))
    print(f"log-area cue: same-pair median {np.median(same_d):.3f} vs diff "
          f"{np.median(diff_d):.3f}  AUC={auc:.3f}  (n_same={len(same_d)})", flush=True)

    # ---- 3. fusion into champion distance ---- #
    def size_dist(q, g):
        za_q = np.array([z.get(it.tracklet_id, (0, 0))[0] for it in q])
        za_g = np.array([z.get(it.tracklet_id, (0, 0))[0] for it in g])
        return np.abs(za_q[:, None] - za_g[None, :])

    report = {"auc_logarea": auc}

    def run(q, g, name):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        base = champ_dist(q, g, embs, sids, cams_qg)
        sd = size_dist(q, g)
        out = {}
        r0 = _score(q, g, base, (1, 5, 10))
        out["base"] = r0
        line = f"  {name:10s} base r1={r0['rank-1']:.3f}"
        for lam in (0.01, 0.03, 0.06):
            r = _score(q, g, base + lam * sd, (1, 5, 10))
            out[f"lam{lam}"] = r
            line += f" | +size({lam}) {r['rank-1']:.3f}"
        print(line, flush=True)
        report[name] = out
        return out

    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in sids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in sids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    run(q1, g1, "P1")
    dm = {}
    for X in sorted({by_tid[t].camera for t in sids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in sids if by_tid[t].camera != X]
        gset = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in sids
             if by_tid[t].camera == X and gt[t] in gset]
        if q:
            dm[X] = run(q, g, f"q_{X}")
    for lam_key in ("base", "lam0.01", "lam0.03", "lam0.06"):
        vals = [dm[X][lam_key]["rank-1"] for X in dm]
        print(f"  dorsal mean [{lam_key}]: {np.mean(vals):.3f}", flush=True)
    id_cams = defaultdict(set)
    for t in sids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in sids if gt[t] in multi]
    run(items, items, "P2")

    with open("artifacts2/sizecue_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/sizecue_diag_v1.json", flush=True)


if __name__ == "__main__":
    main()
