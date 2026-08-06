"""Phase-1 driver: build manifest -> tracklets -> mined pairs -> splits -> scramble.

Examples
--------
# Fast path (no pixels): Tier-1 + Tier-2 anchors + splits + scramble
python build_phase1.py --listing 2025Sep18.listing.txt --out artifacts

# Full Tier-2/Tier-3 with colour-histogram features (extracts needed crops once)
python build_phase1.py --listing 2025Sep18.listing.txt --out artifacts \
    --tar 2025Sep18.tar.gz --work _crops --features color

All outputs are written with never-overwrite versioned names (``_v1`` ...).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import json
from pathlib import Path

from cowreid import (CachedFeatureStore, CameraTopology, ColorHistogramExtractor,
                     DinoV2Extractor, ImageLoader, LabelScrambler, Manifest,
                     SplitGenerator, TemporalSyncPairMiner, build_tracklets,
                     extract_paths)
from cowreid.io_utils import write_csv, write_json


def build_feature_store(args, paths_needed):
    if args.features == "none":
        return None
    if args.tar is None and args.root is None:
        raise SystemExit("--features requires --tar (to extract crops) or --root")
    if args.tar is not None and args.root is None:
        print(f"[features] extracting {len(paths_needed)} crops -> {args.work}")
        extract_paths(args.tar, paths_needed, args.work)
        loader = ImageLoader(root=args.work, tar_path=args.tar)
    else:
        loader = ImageLoader(root=args.root, tar_path=args.tar)
    extractor = ColorHistogramExtractor() if args.features == "color" else DinoV2Extractor()
    return CachedFeatureStore(loader, extractor)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--listing", help="text file of archive member names")
    src.add_argument("--root", help="extracted dataset directory")
    ap.add_argument("--tar", help="tarball (for on-demand crop extraction)")
    ap.add_argument("--work", default="_crops", help="dir to extract crops into")
    ap.add_argument("--out", default="artifacts", help="output directory")
    ap.add_argument("--features", choices=["none", "color", "dino"], default="none")
    ap.add_argument("--max-gap", type=int, default=2, help="tracklet split gap (s)")
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier2-refine", type=int, default=2)
    ap.add_argument("--tier2-min-conf", type=float, default=0.5)
    ap.add_argument("--overlap-threshold", type=float, default=0.02,
                    help="topology weight above which a camera pair counts as overlapping")
    ap.add_argument("--tier3-min-votes", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. manifest --------------------------------------------------------- #
    if args.listing:
        manifest = Manifest.from_listing_file(args.listing)
    else:
        manifest = Manifest.from_dir(args.root)
    stats = manifest.stats()
    print("[manifest]", json.dumps(stats, ensure_ascii=False))
    write_json(out / "manifest_stats.json", stats)
    manifest.to_csv(out / "manifest.csv")

    # 2. tracklets -------------------------------------------------------- #
    tracklets = build_tracklets(manifest, max_gap_s=args.max_gap)
    print(f"[tracklets] {len(tracklets)} tracklets "
          f"(mean {sum(t.n_frames for t in tracklets) / len(tracklets):.1f} frames)")

    # 3a. camera topology (oracle for analysis; estimate if features available) -- #
    topo = CameraTopology.from_gt(manifest)
    write_csv(out / "topology_gt.csv", topo.as_rows(),
              header=["camA", "camB", "weight", "source"])
    overlap = sorted(tuple(sorted(p)) for p in topo.overlapping_pairs(args.overlap_threshold))
    print(f"[topology] overlapping pairs (>= {args.overlap_threshold}): {overlap}")

    # 3b. pair mining ----------------------------------------------------- #
    # Tier-2 positives are only sought on overlapping pairs -> only those crops
    miner_probe = TemporalSyncPairMiner(manifest, tracklets)
    allowed = miner_probe._allowed_pairs(topo, args.overlap_threshold, want_overlap=True)
    needed = {s.path for _t, ca, cb, sa, sb in miner_probe._cooccurrence_bags()
              if frozenset((ca, cb)) in allowed for s in sa + sb}
    fs = build_feature_store(args, sorted(needed))

    miner = TemporalSyncPairMiner(manifest, tracklets, feature_store=fs)

    t1 = miner.mine_tier1(mode="adjacent", window_s=args.max_gap)
    write_csv(out / "pairs_tier1.csv", (p.as_row() for p in t1),
              header=["tier", "path_a", "path_b", "weight", "source",
                      "meta_camera", "meta_tracklet", "meta_dt"])
    print(f"[tier1] {len(t1)} within-camera positive pairs")

    t2_pos = []
    if fs is not None:
        t2_pos = miner.mine_tier2(topology=topo, overlap_threshold=args.overlap_threshold,
                                  n_refine_iters=args.tier2_refine,
                                  min_confidence=args.tier2_min_conf, verbose=True)
    else:
        print("[tier2] no features (--features none) -> skipping cross-view positives")
    t2_neg = miner.mine_tier2_negatives(topo, overlap_threshold=args.overlap_threshold)
    write_csv(out / "pairs_tier2.csv",
              (p.as_row() for p in (t2_pos + t2_neg)),
              header=["tier", "path_a", "path_b", "weight", "source",
                      "meta_t", "meta_camA", "meta_camB", "meta_dist"])
    print(f"[tier2] {len(t2_pos)} cross-view positives (overlap pairs) + "
          f"{len(t2_neg)} hard negatives (non-overlap pairs)")

    clusters, graph = miner.mine_tier3(t2_pos, min_votes=args.tier3_min_votes)
    write_csv(out / "tier3_clusters.csv",
              ({"pseudo_id": c.pseudo_id, "tracklet_id": tid}
               for c in clusters for tid in c.tracklet_ids),
              header=["pseudo_id", "tracklet_id"])
    write_json(out / "tier3_graph.json", graph)
    cluster_eval = miner.evaluate_clusters(clusters)
    print("[tier3]", json.dumps(cluster_eval))
    write_json(out / "tier3_eval.json", cluster_eval)

    # 4. splits ----------------------------------------------------------- #
    gen = SplitGenerator(tracklets, seed=args.seed)
    random_split = gen.make_split(disjoint_by="identity", name="random_open_set")
    loco_split = gen.make_leave_camera_out(holdout=args.holdout_camera)
    for sp in (random_split, loco_split):
        write_json(out / f"split_{sp.name}.json", sp.to_dict())
        print(f"[split:{sp.name}] train={len(sp.train)} val={len(sp.val)} "
              f"test={len(sp.test)} query={len(sp.query)} gallery={len(sp.gallery)}")

    # 5. scrambled label controls ---------------------------------------- #
    scr = LabelScrambler(tracklets, seed=args.seed)
    LabelScrambler.to_csv(scr.scramble(mode="permute", noise_rate=1.0),
                          out / "labels_scrambled_permute.csv")
    for p in (0.2, 0.5):
        LabelScrambler.to_csv(scr.scramble(mode="symmetric", noise_rate=p),
                              out / f"labels_noise_sym{int(p * 100)}.csv")
    print("[scramble] wrote permute + symmetric(0.2,0.5) controls")
    print(f"\nDone. Artifacts in {out.resolve()}")


if __name__ == "__main__":
    main()
