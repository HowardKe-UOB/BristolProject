"""Lever: MULTI-BACKBONE feature fusion (label-free, CPU).

Different backbones make different mistakes, so fusing their rankings can beat
either alone. We fuse the strong ViT-B unsupervised feature (feat768, the current
0.706 champion) with the ViT-S DINOv2 feature from the cached clip features
(`dino_clip_feats_v1.npz`, per-tracklet clip mean). Tested by reciprocal-rank
fusion (RRF) and by concatenation, each with the champion CC/PCAW recipe.

    python new_levers3.py
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
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

HOLD = "66.130"
RANKS = (1, 5, 10)


def champion_dist(q, g, Qf, Gf, cams):
    """The current best recipe RRF(CC, PCAW, CC-RR) on a single feature set."""
    X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
    Qcc, Gcc = cc[:len(q)], cc[len(q):]
    Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
    return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                dist_rerank(Qcc, Gcc, cams, k1=30, k2=6)], k=20), \
        {"CC": dist_cosine(Qcc, Gcc), "PCAW": dist_cosine(Qw, Gw),
         "CC-RR": dist_rerank(Qcc, Gcc, cams, k1=30, k2=6)}


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS)
    dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:34s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
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

    # ViT-B unsupervised champion feature
    db = np.load("_vitb_unsup_emb_v1.npz", allow_pickle=True)
    bidx = {t: i for i, t in enumerate(db["ids"])}
    Bf = db["feat768"]
    QB = np.stack([Bf[bidx[it.tracklet_id]] for it in q])
    GB = np.stack([Bf[bidx[it.tracklet_id]] for it in g])

    # ViT-S DINOv2 cached-clip feature (per-tracklet mean, L2)
    ds = np.load("dino_clip_feats_v1.npz", allow_pickle=True)
    sfeat = {t: (v.mean(0) / (np.linalg.norm(v.mean(0)) + 1e-12))
             for t, v in zip(ds["ids"], ds["clips"])}
    QS = np.stack([sfeat[it.tracklet_id] for it in q])
    GS = np.stack([sfeat[it.tracklet_id] for it in g])

    report = {}
    print(f"leave-out {HOLD}  |Q|={len(q)} |G|={len(g)}\n", flush=True)

    print("-- single backbones (champion recipe) --")
    dB, _ = champion_dist(q, g, QB, GB, cams)
    show("ViT-B champion", q, g, dB, mask, report)
    dS, _ = champion_dist(q, g, QS, GS, cams)
    show("ViT-S(frozen) champion", q, g, dS, mask, report)

    print("\n-- RRF fusion of the two backbones' champion rankings --")
    for w in (1, 2, 3):
        show(f"RRF(ViT-B x{w}, ViT-S)", q, g, rrf([dB] * w + [dS]), mask, report)

    print("\n-- concatenated feature, then champion recipe --")
    QC = np.concatenate([QB / np.linalg.norm(QB, axis=1, keepdims=True),
                         QS / np.linalg.norm(QS, axis=1, keepdims=True)], axis=1)
    GC = np.concatenate([GB / np.linalg.norm(GB, axis=1, keepdims=True),
                         GS / np.linalg.norm(GS, axis=1, keepdims=True)], axis=1)
    dC, _ = champion_dist(q, g, QC, GC, cams)
    show("concat(ViT-B,ViT-S) champion", q, g, dC, mask, report)
    # weighted concat: down-weight the weaker ViT-S
    for a in (0.3, 0.5):
        QCa = np.concatenate([QB / np.linalg.norm(QB, axis=1, keepdims=True),
                              a * QS / np.linalg.norm(QS, axis=1, keepdims=True)], axis=1)
        GCa = np.concatenate([GB / np.linalg.norm(GB, axis=1, keepdims=True),
                              a * GS / np.linalg.norm(GS, axis=1, keepdims=True)], axis=1)
        dCa, _ = champion_dist(q, g, QCa, GCa, cams)
        show(f"concat(ViT-B, {a}*ViT-S) champion", q, g, dCa, mask, report)

    with open("artifacts2/new_levers3_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nchampion ref: r1=0.706 r5=0.859 mAP=0.423")
    print("saved artifacts2/new_levers3_v1.json")


if __name__ == "__main__":
    main()
