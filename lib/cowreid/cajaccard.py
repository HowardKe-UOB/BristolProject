"""CA-Jaccard: camera-aware Jaccard distance + DBSCAN clustering (CVPR 2024,
Chen et al.), adapted for our cross-camera cattle setting.

Why: standard k-reciprocal Jaccard distance is unreliable under large camera
variation because intra-camera samples dominate each k-reciprocal neighbour set,
excluding inter-camera positives -> the same cow gets split across cameras
(exactly our over-segmentation bottleneck). CA-Jaccard adds inter-camera
reciprocal neighbours so cross-camera positives enter the neighbour sets, making
the distance (and the resulting DBSCAN clusters) merge the same cow across views.

This is the standard re-ranking encoding (Zhong et al. 2017) with a camera-aware
neighbour construction, fed to DBSCAN(metric="precomputed"). Topology cannot-link
pairs are injected as max distance so they are never directly linked.
"""
from __future__ import annotations

import numpy as np


def _k_reciprocal(initial_rank: np.ndarray, i: int, k1: int) -> np.ndarray:
    forward = initial_rank[i, : k1 + 1]
    backward = initial_rank[forward, : k1 + 1]
    valid = np.where((backward == i).any(axis=1))[0]
    return forward[valid]


def ca_jaccard_distance(feat: np.ndarray, cameras, k1: int = 20, k2: int = 6,
                        camera_aware: bool = True, lambda_value: float = 0.0) -> np.ndarray:
    """Return an (N, N) camera-aware Jaccard distance matrix in [0, 1]."""
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
    N = len(feat)
    original_dist = np.clip(1.0 - feat @ feat.T, 0.0, None).astype(np.float32)
    initial_rank = np.argsort(original_dist, axis=1)
    cameras = np.asarray(cameras)

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        kr = _k_reciprocal(initial_rank, i, k1)
        kr_exp = list(kr)
        for j in kr:                                   # local expansion (half-k1)
            cand = _k_reciprocal(initial_rank, j, int(round(k1 / 2)))
            if len(cand) and len(np.intersect1d(cand, kr)) > (2.0 / 3) * len(cand):
                kr_exp.extend(cand.tolist())
        kr_exp = np.unique(kr_exp)
        if camera_aware:                               # add inter-camera neighbours
            inter = np.where(cameras != cameras[i])[0]
            if len(inter):
                inter_sorted = inter[np.argsort(original_dist[i, inter])]
                kr_exp = np.unique(np.concatenate([kr_exp, inter_sorted[: max(1, k1 // 2)]]))
        w = np.exp(-original_dist[i, kr_exp])
        V[i, kr_exp] = (w / w.sum()).astype(np.float32)

    if k2 > 1:                                         # local query expansion
        V = np.stack([V[initial_rank[i, :k2]].mean(axis=0) for i in range(N)])

    jacc = np.zeros((N, N), dtype=np.float32)
    for i in range(N):                                 # Jaccard via min/max
        sm = np.minimum(V[i][None, :], V).sum(axis=1)
        sx = np.maximum(V[i][None, :], V).sum(axis=1)
        jacc[i] = 1.0 - sm / (sx + 1e-12)
    jacc = 0.5 * (jacc + jacc.T)
    if lambda_value > 0:
        jacc = jacc * (1 - lambda_value) + original_dist * lambda_value
    np.fill_diagonal(jacc, 0.0)
    return jacc


def dbscan_cluster(ids, feat: np.ndarray, cameras, eps: float = 0.6,
                   min_samples: int = 2, k1: int = 20, k2: int = 6,
                   camera_aware: bool = True, cannot_link=None) -> dict:
    """Cluster with DBSCAN on the CA-Jaccard distance. Returns {id -> label}; DBSCAN
    outliers become their own singleton clusters. cannot_link pairs are pushed to
    max distance so they are not directly density-linked."""
    from sklearn.cluster import DBSCAN

    D = ca_jaccard_distance(feat, cameras, k1, k2, camera_aware)
    if cannot_link:
        pos = {t: i for i, t in enumerate(ids)}
        for p in cannot_link:
            a, b = tuple(p)
            if a in pos and b in pos:
                D[pos[a], pos[b]] = D[pos[b], pos[a]] = 1.0
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
    out, nxt = {}, (int(labels.max()) + 1 if labels.max() >= 0 else 0)
    for t, l in zip(ids, labels):
        if l < 0:
            out[t] = nxt; nxt += 1
        else:
            out[t] = int(l)
    return out


def num_clusters(labels: dict) -> int:
    return len(set(labels.values()))
