"""cowreid -- Phase 1 utilities for self-supervised multi-camera cattle Re-ID.

Pipeline overview::

    Manifest                      # parse 2025Sep18/<cam>/<id>/<ts>.jpeg
      -> build_tracklets          # intra-camera contiguous runs (tracker proxy)
      -> TemporalSyncPairMiner     # Tier1 / Tier2 (OT) / Tier3 (cross-cam clusters)
      -> SplitGenerator            # tracklet-disjoint, cross-camera, leave-cam-out
      -> LabelScrambler            # control experiment

GT identities are used only for evaluation and for the scrambling control; the
self-supervised miner never consumes them (the sole exception is the documented
tracker-proxy in tracklet construction).
"""
from .features import (CachedFeatureStore, ColorHistogramExtractor,
                       DinoV2Extractor, FeatureExtractor)
from .io_utils import ImageLoader, extract_paths, versioned_path
from .manifest import Manifest, Sample, parse_path
from .pair_miner import Pair, TemporalSyncPairMiner, TrackletCluster
from .splits import LabelScrambler, Split, SplitGenerator
from .topology import CameraTopology
from .tracklets import Tracklet, TrackletIndex, build_tracklets

__all__ = [
    "Manifest", "Sample", "parse_path",
    "build_tracklets", "Tracklet", "TrackletIndex",
    "TemporalSyncPairMiner", "Pair", "TrackletCluster",
    "SplitGenerator", "Split", "LabelScrambler", "CameraTopology",
    "ImageLoader", "extract_paths", "versioned_path",
    "FeatureExtractor", "ColorHistogramExtractor", "DinoV2Extractor",
    "CachedFeatureStore",
]
