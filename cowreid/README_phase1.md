# Phase 1 — Temporal-Sync Pair Miner & Split Generator

Self-supervised multi-camera cattle Re-ID tooling for the `2025Sep18` dataset
(7 cameras, 98 identities, 124,145 crops, one ~2.2 h afternoon session).
Ground-truth identities are used **only** for evaluation and for the scrambling
control — the miner never consumes them (sole exception: a documented
single-camera tracker *proxy* when forming intra-camera tracklets).

## Install / run

```bash
# Fast path: Tier-1 + topology + hard negatives + splits + scramble (no pixels)
python build_phase1.py --listing 2025Sep18.listing.txt --out artifacts

# With cross-view positive mining (needs pixels + features)
python build_phase1.py --listing 2025Sep18.listing.txt --out artifacts \
    --tar 2025Sep18.tar.gz --work _crops --features color   # or: --features dino
```
Only `numpy` is required; `Pillow` for pixels; `torch`+`timm` only for `DinoV2Extractor`.
All outputs use never-overwrite versioned names (`_v1`, `_v2`, ...).

## ⚠️ Key empirical finding (changes the design)

The 7 cameras do **not** share one field of view. Validated against GT:

| Quantity | Value |
|---|---|
| Same-instant cross-camera pair is the *same* cow (base rate) | **1.66 %** |
| "1-cow-vs-1-cow slot" is the same cow (the assumed anchor) | **0.5 %** → seed is invalid, **removed** |
| OT cross-view *positive* precision, best overlap pair, colour feats | ~16–25 % (needs a strong backbone) |
| Camera pairs with non-trivial overlap | 7 of 21; the rest ≈ 0 % |
| Same-instant pair in a *non-overlap* pair is a *different* cow | **99.86 %** (→ hard negatives) |

So temporal sync's reliable contribution is the **inverse** of the original plan:
a label-free **camera-overlap topology** + **~625 k near-perfect hard negatives**.
Cross-view *positives* are deferred until a trained backbone (DINOv2) exists, then
re-mined on overlapping pairs only (iterate mining ↔ training).

## Modules

| File | Role |
|---|---|
| `manifest.py` | parse layout → `Sample`; filtering, co-occurrence index, stats |
| `tracklets.py` | `build_tracklets` (intra-camera contiguous runs; tracker proxy), `TrackletIndex` |
| `topology.py` | `CameraTopology.from_gt` (oracle) / `.estimate` (label-free overlap graph) |
| `sinkhorn.py` | entropic OT + `match_with_dustbin` (reject option) |
| `features.py` | `ColorHistogramExtractor`, `DinoV2Extractor`, `CachedFeatureStore` |
| `pair_miner.py` | `TemporalSyncPairMiner`: Tier-1 / Tier-2 positives (overlap-gated) + negatives / Tier-3 clusters |
| `splits.py` | `SplitGenerator` (tracklet-disjoint, open-set, leave-camera-out), `LabelScrambler` |
| `io_utils.py` | versioned writers, tar extraction, `ImageLoader` |

## Interfaces the training pipeline consumes

* `pairs_tier1.csv` — within-camera positives `(path_a, path_b, weight)`.
* `pairs_tier2.csv` — `source=ot` cross-view positives (overlap pairs) and
  `source=neg` hard negatives (non-overlap pairs).
* `tier3_clusters.csv` — `tracklet_id → pseudo_id` (cross-camera pseudo-labels).
* `split_*.json` — train/val/test tracklet ids + cross-camera query/gallery.
* `labels_*.csv` — scrambled-label controls (`permute`, `symmetric` noise rates).
* `topology_gt.csv` — camera-pair overlap weights.
