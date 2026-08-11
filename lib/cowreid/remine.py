"""Re-mining hook: use the trained encoder to refresh cross-view positives.

Closes the mining<->training loop. The encoder is adapted to the Phase-1
``FeatureExtractor`` protocol so ``TemporalSyncPairMiner.mine_tier2`` (and
``CameraTopology.estimate``) can be re-run on the overlapping camera pairs with
learned features -- the point at which temporal cross-view positives are expected
to become reliable.
"""
from __future__ import annotations

import numpy as np
import torch

from .features import CachedFeatureStore
from .io_utils import ImageLoader
from .manifest import Manifest
from .pair_miner import TemporalSyncPairMiner
from .topology import CameraTopology


class EncoderFeatureExtractor:
    """Wrap a ``VideoReIDEncoder`` as a per-image Phase-1 feature extractor."""

    def __init__(self, encoder, transform, device: str = "cpu"):
        self.encoder = encoder.eval().to(device)
        self.transform = transform
        self.device = device

    @torch.no_grad()
    def extract(self, image) -> np.ndarray:
        x = self.transform(image).unsqueeze(0).to(self.device)   # (1, C, H, W) -> T=1
        emb = self.encoder(x)["embed"][0].cpu().numpy()
        return emb.astype(np.float32)


def remine_overlap_positives(encoder, manifest: Manifest, tracklets,
                             topology: CameraTopology, loader: ImageLoader, transform,
                             device: str = "cpu", overlap_threshold: float = 0.02,
                             min_confidence: float = 0.5, n_refine_iters: int = 2,
                             verbose: bool = True):
    """Re-mine Tier-2 cross-view positives on overlapping pairs with learned features.

    Returns the list of Tier-2 positive ``Pair`` objects (source='ot')."""
    fs = CachedFeatureStore(loader, EncoderFeatureExtractor(encoder, transform, device))
    miner = TemporalSyncPairMiner(manifest, tracklets, feature_store=fs)
    return miner.mine_tier2(topology=topology, overlap_threshold=overlap_threshold,
                            min_confidence=min_confidence, n_refine_iters=n_refine_iters,
                            verbose=verbose)
