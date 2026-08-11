"""Crop appearance features for Tier-2 matching.

The pair miner only needs a function ``paths -> (N, D) array``. Any embedding works.
Two implementations are provided:

* :class:`ColorHistogramExtractor` -- dependency-light (Pillow + NumPy). Good enough
  to bootstrap Tier-2 matching before any model exists. Coat-pattern colour stats
  are surprisingly discriminative for Holsteins at a fixed instant.
* :class:`DinoV2Extractor` -- optional, requires ``torch``/``timm``; the recommended
  feature once you have a backbone. Plug your trained encoder here to *iterate*
  mining <-> training (re-mine pairs with improved features each round).

:class:`CachedFeatureStore` wires a loader + extractor together and caches vectors
by path so repeated mining passes are cheap.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .io_utils import ImageLoader


class FeatureExtractor(Protocol):
    """Maps a PIL image to a 1-D float32 feature vector."""

    def extract(self, image) -> np.ndarray: ...


class ColorHistogramExtractor:
    """Concatenated per-channel histograms in RGB + HSV, L2-normalised."""

    def __init__(self, bins: int = 8):
        self.bins = bins

    def extract(self, image) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
        feats = []
        for arr in (rgb, hsv):
            for c in range(3):
                h, _ = np.histogram(arr[..., c], bins=self.bins, range=(0.0, 1.0))
                feats.append(h.astype(np.float32))
        v = np.concatenate(feats)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class DinoV2Extractor:
    """Frozen DINOv2 CLS embedding (optional; requires torch + timm)."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m",
                 device: str | None = None, image_size: int | None = None):
        import timm
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(self.device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        if image_size is not None:  # default: use the model's native size (518 for DINOv2/14)
            cfg["input_size"] = (3, image_size, image_size)
        self.transform = timm.data.create_transform(**cfg)

    def extract(self, image) -> np.ndarray:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model(x).squeeze(0).float().cpu().numpy()
        n = np.linalg.norm(feat)
        return (feat / n).astype(np.float32) if n > 0 else feat.astype(np.float32)


class CachedFeatureStore:
    """Lazily extracts and caches features keyed by relative path."""

    def __init__(self, loader: ImageLoader, extractor: FeatureExtractor):
        self.loader = loader
        self.extractor = extractor
        self._cache: dict[str, np.ndarray] = {}

    def precompute(self, paths: Sequence[str], verbose: bool = False) -> None:
        todo = [p for p in dict.fromkeys(paths) if p not in self._cache]
        for k, p in enumerate(todo):
            self._cache[p] = self.extractor.extract(self.loader.load(p))
            if verbose and (k + 1) % 2000 == 0:
                print(f"  features {k + 1}/{len(todo)}")

    def get(self, paths: Sequence[str]) -> np.ndarray:
        self.precompute(paths)
        return np.stack([self._cache[p] for p in paths])

    def save(self, path: str) -> None:
        np.savez_compressed(path, keys=np.array(list(self._cache), dtype=object),
                            vals=np.stack(list(self._cache.values())))

    def load_cache(self, path: str) -> None:
        d = np.load(path, allow_pickle=True)
        self._cache.update({k: v for k, v in zip(d["keys"], d["vals"])})
