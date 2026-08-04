"""Validate the inference-time spatio-temporal mask (cowreid/st_inference.py).

CPU-only, no training. Three questions:
  1. SAFETY  -- does the mask ever remove a TRUE cross-camera match? (GT used for
     measurement only; expect ~0 given the 99.9% reliability of the same-instant
     non-overlap negative signal.)
  2. POWER   -- what fraction of the gallery does it prune per query?
  3. PAYOFF  -- does it lift retrieval on the frozen DINOv2 features (leave-out
     66.130), raw cosine and CA-Jaccard re-ranked?

    python st_validate.py --listing 2025Sep18.listing.txt
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, ReIDEvaluator, evaluate_rerank
from cowreid.st_inference import (build_st_mask, evaluate_rerank_st, evaluate_st,
                                  mask_oracle_check)
from cowreid.tracklets import TrackletIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--overlap-thr", type=float, default=0.02)
    ap.add_argument("--margins", type=int, nargs="+", default=[0, 3, 10])
    ap.add_argument("--out", default="artifacts2/st_mask_validation_v1.json")
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
    print(f"protocol: leave-out {hold}  |Q|={len(query)} |G|={len(gallery)}", flush=True)

    report = {"protocol": f"leave-out {hold}", "n_query": len(query),
              "n_gallery": len(gallery), "overlap_thr": args.overlap_thr,
              "margins": {}}

    # ---- 1+2: safety & power at several margins ---------------------------- #
    for m in args.margins:
        mask = build_st_mask(query, gallery, index, topo, args.overlap_thr, margin=m)
        chk = mask_oracle_check(query, gallery, mask)
        report["margins"][m] = chk
        print(f"[margin={m:2d}s] gallery pruned/query={chk['mean_gallery_masked']:.1%}  "
              f"true matches masked={chk['true_matches_masked']}/{chk['true_matches']} "
              f"({chk['true_masked_rate']:.3%})  dead queries={chk['queries_all_true_masked']}",
              flush=True)

    # ---- 3: payoff on frozen features -------------------------------------- #
    d = np.load(args.cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    emb = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    ev = ReIDEvaluator(ranks=(1, 5, 10))
    base = ev.evaluate(query, gallery, emb)
    rr = evaluate_rerank(query, gallery, emb)
    print(f"\nfrozen cosine      : mAP={base['mAP']:.3f} r1={base['rank-1']:.3f} "
          f"r5={base['rank-5']:.3f} r10={base['rank-10']:.3f}", flush=True)
    print(f"frozen rerank      : mAP={rr['mAP']:.3f} r1={rr['rank-1']:.3f} "
          f"r5={rr['rank-5']:.3f} r10={rr['rank-10']:.3f}", flush=True)
    report["frozen_cosine"], report["frozen_rerank"] = base, rr

    for m in args.margins:
        st = evaluate_st(query, gallery, emb, index, topo, margin=m,
                         overlap_thr=args.overlap_thr)
        st_rr = evaluate_rerank_st(query, gallery, emb, index, topo, margin=m,
                                   overlap_thr=args.overlap_thr)
        print(f"frozen cosine +ST(m={m:2d}): mAP={st['mAP']:.3f} r1={st['rank-1']:.3f} "
              f"r5={st['rank-5']:.3f} r10={st['rank-10']:.3f}", flush=True)
        print(f"frozen rerank +ST(m={m:2d}): mAP={st_rr['mAP']:.3f} r1={st_rr['rank-1']:.3f} "
              f"r5={st_rr['rank-5']:.3f} r10={st_rr['rank-10']:.3f}", flush=True)
        report[f"frozen_cosine_st_m{m}"], report[f"frozen_rerank_st_m{m}"] = st, st_rr

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
