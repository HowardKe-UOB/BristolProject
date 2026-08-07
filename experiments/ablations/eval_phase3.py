"""Phase-3 baseline: embed test tracklets with a FROZEN feature extractor (no
training) and report cross-camera Re-ID accuracy. This is the honest lower-bound
("how far do off-the-shelf features get us?") that a trained encoder must beat.

    python eval_phase3.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz \
        --features color            # or: --features dino   (downloads DINOv2)

Protocols:
  * full      -- all multi-camera identities, all-vs-all cross-camera (stable headline)
  * leave_out -- train cameras as gallery, the held-out viewpoint as query (cross-view DG)
  * open_set  -- the Phase-1 identity-disjoint test split (few queries, noisy)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse

import numpy as np

from cowreid import (CachedFeatureStore, ColorHistogramExtractor, DinoV2Extractor,
                     ImageLoader, Manifest, SplitGenerator, build_tracklets, extract_paths)
from cowreid.eval import EvalItem, ReIDEvaluator, build_full_cross_camera
from cowreid.io_utils import write_json
from cowreid.tracklets import TrackletIndex


def sample_frames(tracklet, k: int) -> list[str]:
    paths = tracklet.paths
    idx = np.linspace(0, len(paths) - 1, min(k, len(paths))).astype(int)
    return [paths[i] for i in idx]


def embed_tracklets(tids, index, fs, k):
    paths_per = {t: sample_frames(index[t], k) for t in tids}
    fs.precompute(sorted({p for ps in paths_per.values() for p in ps}), verbose=True)
    emb = {}
    for t, ps in paths_per.items():
        v = fs.get(ps).mean(axis=0)
        emb[t] = v / (np.linalg.norm(v) + 1e-12)
    return emb


def to_items(split_field):
    return [EvalItem(d["tracklet_id"], d["identity"], d["camera"]) for d in split_field]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_eval")
    # "dino" is ViT-S/14 (the phase-1 seed extractor); "dinob" is ViT-B/14, the very
    # backbone the training ladder adapts -- its frozen score is the floor quoted against
    # the supervised 0.969 in Chapter 5.
    ap.add_argument("--features", choices=["color", "dino", "dinob", "random"], default="color")
    ap.add_argument("--frames-per-tracklet", type=int, default=6)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--out", default="artifacts2/eval_phase3.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    gen = SplitGenerator(tracklets, seed=args.seed)

    # protocols
    full_q, full_g = build_full_cross_camera(tracklets)
    loco = gen.make_leave_camera_out(holdout=args.holdout_camera)
    openset = gen.make_split(disjoint_by="identity", name="open_set")
    protocols = {
        "full_cross_camera": (full_q, full_g),
        f"leave_out_{args.holdout_camera}": (to_items(loco.query), to_items(loco.gallery)),
        "open_set_split": (to_items(openset.query), to_items(openset.gallery)),
    }

    # embed every tracklet referenced by any protocol
    needed_tids = sorted({it.tracklet_id
                          for q, g in protocols.values() for it in (q + g)})
    print(f"[embed] {len(needed_tids)} tracklets x {args.frames_per_tracklet} frames "
          f"with '{args.features}' features")

    if args.features == "random":
        rng = np.random.default_rng(args.seed)
        emb = {t: rng.standard_normal(128) for t in needed_tids}
    else:
        needed_paths = sorted({p for t in needed_tids
                               for p in sample_frames(index[t], args.frames_per_tracklet)})
        extract_paths(args.tar, needed_paths, args.work)
        extractor = (ColorHistogramExtractor() if args.features == "color"
                     else DinoV2Extractor(model_name="vit_base_patch14_dinov2.lvd142m")
                     if args.features == "dinob"
                     else DinoV2Extractor())
        fs = CachedFeatureStore(ImageLoader(root=args.work, tar_path=args.tar), extractor)
        emb = embed_tracklets(needed_tids, index, fs, args.frames_per_tracklet)

    evaluator = ReIDEvaluator(ranks=(1, 5, 10))
    results = {}
    print(f"\n=== Cross-camera Re-ID ({args.features} features, FROZEN / no training) ===")
    for name, (q, g) in protocols.items():
        r = evaluator.evaluate(q, g, emb)
        results[name] = r
        print(f"  {name:24s}  mAP={r['mAP']:.3f}  rank-1={r['rank-1']:.3f}  "
              f"rank-5={r['rank-5']:.3f}  (q={r['n_query_scored']}/{r['n_query_total']}, "
              f"g={r['n_gallery']})")
    write_json(args.out, {"features": args.features,
                          "frames_per_tracklet": args.frames_per_tracklet,
                          "results": results})


if __name__ == "__main__":
    main()
