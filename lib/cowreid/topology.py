"""Camera-overlap topology.

A central empirical finding on this dataset: the 7 cameras do NOT share one field
of view. They form a spatial graph -- a few neighbouring pairs partially overlap,
many pairs never co-observe the same animal. The topology determines where
temporal-sync *positives* are even possible (overlapping pairs) and where
same-instant crops are reliable *negatives* (non-overlapping pairs).

* :meth:`CameraTopology.from_gt` -- oracle overlap weights from GT co-occurrence
  (analysis / figures / sanity only; uses identities).
* :meth:`CameraTopology.estimate` -- label-free estimate from the rate of confident
  mutual-NN appearance matches at co-occurring instants (this is what the method
  would actually use). Correlates with the oracle when features are discriminative.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .features import CachedFeatureStore
from .manifest import Manifest
from .sinkhorn import match_with_dustbin


@dataclass
class CameraTopology:
    weights: dict[tuple[str, str], float] = field(default_factory=dict)  # symmetric, [0,1]
    source: str = "gt"

    def overlapping_pairs(self, threshold: float) -> set[frozenset]:
        return {frozenset(k) for k, w in self.weights.items() if w >= threshold}

    def non_overlapping_pairs(self, threshold: float) -> set[frozenset]:
        return {frozenset(k) for k, w in self.weights.items() if w < threshold}

    def is_overlap(self, camA: str, camB: str, threshold: float) -> bool:
        return self.weights.get(tuple(sorted((camA, camB))), 0.0) >= threshold

    def as_rows(self) -> list[dict]:
        return [{"camA": a, "camB": b, "weight": round(w, 5), "source": self.source}
                for (a, b), w in sorted(self.weights.items(), key=lambda kv: -kv[1])]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_gt(cls, manifest: Manifest) -> "CameraTopology":
        """Oracle: fraction of co-occurring cross-camera crop pairs that share an id."""
        same = defaultdict(int)
        tot = defaultdict(int)
        for _t, cam_map in manifest.by_timestamp().items():
            cams = [c for c in cam_map if cam_map[c]]
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    a, b = sorted((cams[i], cams[j]))
                    la = [s.gt_label for s in cam_map[cams[i]]]
                    lb = [s.gt_label for s in cam_map[cams[j]]]
                    sb = set(lb)
                    s = sum(1 for x in la for y in lb if x == y)
                    same[(a, b)] += s
                    tot[(a, b)] += len(la) * len(lb)
        weights = {k: (same[k] / tot[k] if tot[k] else 0.0) for k in tot}
        return cls(weights=weights, source="gt")

    @classmethod
    def estimate(cls, manifest: Manifest, feature_store: CachedFeatureStore,
                 eps: float = 0.1, min_conf: float = 0.5,
                 max_slots_per_pair: int = 400, seed: int = 0) -> "CameraTopology":
        """Label-free: per camera pair, the rate of confident dustbin-OT matches at
        co-occurring instants, normalised by the number of slots sampled."""
        rng = np.random.default_rng(seed)
        slots: dict[tuple[str, str], list] = defaultdict(list)
        for t, cam_map in manifest.by_timestamp().items():
            cams = [c for c in cam_map if cam_map[c]]
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    a, b = sorted((cams[i], cams[j]))
                    slots[(a, b)].append((cam_map[a], cam_map[b]))

        weights = {}
        for pair, slist in slots.items():
            if len(slist) > max_slots_per_pair:
                idx = rng.choice(len(slist), max_slots_per_pair, replace=False)
                slist = [slist[k] for k in idx]
            n_match = n_slot = 0
            for sa, sb in slist:
                FA = feature_store.get([s.path for s in sa])
                FB = feature_store.get([s.path for s in sb])
                cost = _euclid(FA, FB)
                m = [x for x in match_with_dustbin(cost, eps=eps) if x[2] >= min_conf]
                n_match += len(m)
                n_slot += max(len(sa), len(sb))
            weights[pair] = n_match / n_slot if n_slot else 0.0
        return cls(weights=weights, source="estimate")


def _euclid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d2 = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)
    return np.sqrt(d2)
