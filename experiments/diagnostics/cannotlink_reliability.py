"""Archive the reliability of the same-instant cannot-link signal, the free negative
supervision the whole pipeline leans on. Three numbers the dissertation quotes but no
artifact stored until now:

  * crop-level:     same-second crops on NON-overlapping camera pairs -> how often are
                    they truly different animals?  (thesis: 625,606 pairs, 99.86%)
  * tracklet-level: the cannot-link constraint set at overlap threshold 0.02, cross-camera
                    subset -> reliability of the constraints training actually uses
  * the shortcut:   across ALL camera pairs, how often is "same instant, different camera"
                    the SAME animal? (thesis: 1.66% -- i.e. the naive positive shortcut fails)

    python experiments/diagnostics/cannotlink_reliability.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import argparse
import json
from collections import defaultdict
from itertools import combinations

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import build_cannot_link


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--overlap-threshold", type=float, default=0.02)
    ap.add_argument("--out", default="artifacts2/cannotlink_reliability_v1.json")
    args = ap.parse_args()

    manifest = Manifest.from_listing_file(args.listing)
    topo = CameraTopology.from_gt(manifest)

    def overlapping(a, b):
        return topo.is_overlap(a, b, args.overlap_threshold)

    # ---- crop level: bucket crops by second, pair across cameras ---------------------
    by_t = defaultdict(lambda: defaultdict(list))     # t -> camera -> [gt_label]
    for s in manifest.samples:
        by_t[s.t][s.camera].append(s.gt_label)

    crop = {"nonoverlap_pairs": 0, "nonoverlap_violations": 0,
            "all_pairs": 0, "all_same_identity": 0}
    for cams in by_t.values():
        for (ca, la), (cb, lb) in combinations(cams.items(), 2):
            n = len(la) * len(lb)
            same = sum(1 for x in la for y in lb if x == y)
            crop["all_pairs"] += n
            crop["all_same_identity"] += same
            if not overlapping(ca, cb):
                crop["nonoverlap_pairs"] += n
                crop["nonoverlap_violations"] += same

    crop["reliability"] = 1 - crop["nonoverlap_violations"] / crop["nonoverlap_pairs"]
    crop["shortcut_positive_rate"] = crop["all_same_identity"] / crop["all_pairs"]

    # ---- tracklet level: the constraint set training actually consumes ----------------
    tracklets = build_tracklets(manifest, max_gap_s=2)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam = {t.tracklet_id: t.camera for t in tracklets}
    cl = build_cannot_link(tracklets, topo, args.overlap_threshold)
    pairs = [tuple(p) for p in cl]
    cross = [(a, b) for a, b in pairs if cam[a] != cam[b]]
    viol_all = sum(1 for a, b in pairs if gt[a] == gt[b])
    viol_cross = sum(1 for a, b in cross if gt[a] == gt[b])
    # The dissertation quotes the constraint set restricted to the 997-tracklet evaluation
    # universe (the 82 oblique-only tracklets whose identities never appear elsewhere are
    # excluded): 12,958 constraints at 99.78%.
    gal_ids = {t.gt_label for t in tracklets if t.camera != "66.130"}
    keep = {t.tracklet_id for t in tracklets
            if t.camera != "66.130" or t.gt_label in gal_ids}
    p997 = [(a, b) for a, b in pairs if a in keep and b in keep]
    v997 = sum(1 for a, b in p997 if gt[a] == gt[b])
    trk = {"constraints_997_universe": len(p997), "violations_997": v997,
           "reliability_997": 1 - v997 / len(p997),
           "constraints_total_1079": len(pairs), "violations_total": viol_all,
           "reliability_total": 1 - viol_all / len(pairs),
           "cross_camera_constraints": len(cross), "cross_camera_violations": viol_cross,
           "cross_camera_reliability": 1 - viol_cross / len(cross)}

    out = {"script": "cannotlink_reliability.py", "listing": args.listing,
           "overlap_threshold": args.overlap_threshold,
           "crop_level": crop, "tracklet_level": trk}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"crop level     : {crop['nonoverlap_pairs']:,} non-overlap same-instant pairs, "
          f"{crop['nonoverlap_violations']} violations -> reliability {crop['reliability']:.4%}")
    print(f"shortcut       : same-instant cross-camera pairs are the same animal "
          f"{crop['shortcut_positive_rate']:.2%} of the time ({crop['all_pairs']:,} pairs)")
    print(f"tracklet level : {trk['constraints_997_universe']:,} constraints in the 997 "
          f"universe, {trk['violations_997']} violations -> {trk['reliability_997']:.4%}")
    print(f"                 ({trk['constraints_total_1079']:,} constraints total "
          f"({trk['cross_camera_constraints']:,} cross-camera), "
          f"{trk['violations_total']} violations -> reliability {trk['reliability_total']:.4%} "
          f"(cross-camera {trk['cross_camera_reliability']:.4%})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
