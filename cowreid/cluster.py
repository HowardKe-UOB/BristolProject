"""Cross-camera pseudo-identity assignment from encoder features.

Cluster-Contrast needs a pseudo-label per tracklet, refreshed periodically as the
encoder improves. We cluster tracklet-mean embeddings with a mutual-kNN graph and a
*constrained* union-find that never merges a topology cannot-link pair -- so the
clusters respect "a cow cannot be in two places at once". (The same constrained
union-find backs Tier-3 in :mod:`cowreid.pair_miner`.)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .pair_miner import _ConstrainedUnionFind
from .tracklets import Tracklet
from .topology import CameraTopology


def build_cannot_link(tracklets: Sequence[Tracklet], topology: CameraTopology,
                      overlap_threshold: float = 0.02) -> set[frozenset]:
    """Tracklet pairs that cannot be the same animal:

    (1) same camera, overlapping time; or
    (2) non-overlapping camera pair, overlapping time (different physical locations).
    """
    cl: set[frozenset] = set()
    tl = list(tracklets)
    tl.sort(key=lambda t: t.t_start)
    for i in range(len(tl)):
        ti = tl[i]
        for j in range(i + 1, len(tl)):
            tj = tl[j]
            if tj.t_start > ti.t_end:
                break  # sorted by start; no further overlaps with ti
            if not ti.overlaps_in_time(tj):
                continue
            same_cam = ti.camera == tj.camera
            overlap_cam = topology.is_overlap(ti.camera, tj.camera, overlap_threshold)
            if same_cam or not overlap_cam:
                cl.add(frozenset((ti.tracklet_id, tj.tracklet_id)))
    return cl


class ClusterAssigner:
    """Mutual-kNN + constrained union-find clustering of tracklet embeddings."""

    def __init__(self, sim_threshold: float = 0.6, k: int = 10,
                 min_cluster_size: int = 1):
        self.sim_threshold = sim_threshold
        self.k = k
        self.min_cluster_size = min_cluster_size

    def assign(self, tracklet_ids: Sequence[str], features: np.ndarray,
               cannot_link: set[frozenset] | None = None) -> dict[str, int]:
        """features: (N, D); returns {tracklet_id -> pseudo_label}. Clusters smaller
        than ``min_cluster_size`` are labelled -1 (ignored by the cluster loss)."""
        feats = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        n = len(tracklet_ids)
        sim = feats @ feats.T
        np.fill_diagonal(sim, -1.0)

        # mutual-kNN edges above threshold
        knn = np.argsort(-sim, axis=1)[:, : self.k]
        edges = []
        for i in range(n):
            for j in knn[i]:
                if sim[i, j] >= self.sim_threshold and i in knn[j]:
                    edges.append((sim[i, j], i, int(j)))
        edges.sort(reverse=True)

        cl = cannot_link or set()
        uf = _ConstrainedUnionFind(cl)
        for tid in tracklet_ids:
            uf.add(tid)
        for _s, i, j in edges:
            uf.union(tracklet_ids[i], tracklet_ids[j])

        comp: dict[str, list[str]] = defaultdict(list)
        for tid in tracklet_ids:
            comp[uf.find(tid)].append(tid)

        labels: dict[str, int] = {}
        next_label = 0
        for members in comp.values():
            if len(members) < self.min_cluster_size:
                for tid in members:
                    labels[tid] = -1
            else:
                for tid in members:
                    labels[tid] = next_label
                next_label += 1
        return labels

    @staticmethod
    def num_clusters(labels: dict[str, int]) -> int:
        return len({v for v in labels.values() if v >= 0})
