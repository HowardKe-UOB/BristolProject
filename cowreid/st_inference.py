"""Spatio-temporal constraints at *inference* time (retrieval-side ST masking).

The training pipeline already exploits the camera topology's temporal-sync signal
as hard NEGATIVES (same-instant crops in non-overlapping camera pairs are
different cows, ~99.9% reliable on this dataset). This module applies the same
physical constraint at retrieval: a gallery tracklet whose time interval overlaps
the query tracklet's interval, in a camera whose field of view does NOT overlap
the query camera's, cannot be the same animal (a cow is in one place at a time).
Masked entries get +inf distance before ranking.

This is the tracklet-level analogue of the Time-Overlapping Constraint / spatio-
temporal score fusion used in person Re-ID (TFusion CVPR'18; ST-ReID AAAI'19;
"auxiliary information" Re-ID, arXiv:2205.03124). It consumes NO identity labels
-- only timestamps, camera IDs and the camera-overlap topology -- so it is legal
for the label-free protocol (the topology itself can be estimated label-free,
see :meth:`CameraTopology.estimate`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .eval import EvalItem, _score, _stack
from .topology import CameraTopology
from .tracklets import TrackletIndex

INF = 1e6


def build_st_mask(query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                  index: TrackletIndex, topo: CameraTopology,
                  overlap_thr: float = 0.02, margin: int = 0) -> np.ndarray:
    """(Nq, Ng) bool mask; True = physically impossible pair (exclude from ranking).

    A pair is masked iff (a) the two cameras' fields of view do NOT overlap
    (topology weight < overlap_thr) and (b) the tracklets' time intervals
    intersect for at least ``margin`` seconds (margin=0 masks any intersection,
    incl. a single shared second; raise it to be conservative near interval
    edges / FOV borders).
    """
    q_start = np.array([index[it.tracklet_id].t_start for it in query])
    q_end = np.array([index[it.tracklet_id].t_end for it in query])
    g_start = np.array([index[it.tracklet_id].t_start for it in gallery])
    g_end = np.array([index[it.tracklet_id].t_end for it in gallery])
    q_cam = [it.camera for it in query]
    g_cam = [it.camera for it in gallery]

    cams = sorted(set(q_cam) | set(g_cam))
    non_ov = {(a, b): not topo.is_overlap(a, b, overlap_thr)
              for a in cams for b in cams if a != b}
    cam_mask = np.array([[non_ov.get((qc, gc), False) for gc in g_cam] for qc in q_cam])

    inter = (np.minimum(q_end[:, None], g_end[None, :])
             - np.maximum(q_start[:, None], g_start[None, :]))
    return cam_mask & (inter >= margin)


def mask_oracle_check(query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                      mask: np.ndarray) -> dict:
    """GT diagnostic (labels used for *measurement only*): how often would the ST
    mask remove a TRUE cross-camera match, and how much of the gallery does it
    prune on average?"""
    g_ids = np.array([it.identity for it in gallery])
    g_cams = np.array([it.camera for it in gallery])
    n_true = n_true_masked = n_dead = 0
    for i, qi in enumerate(query):
        true = (g_ids == qi.identity) & (g_cams != qi.camera)
        n_true += int(true.sum())
        n_true_masked += int((true & mask[i]).sum())
        if true.any() and (true & ~mask[i]).sum() == 0:
            n_dead += 1                      # every true match masked -> query unanswerable
    return {"mean_gallery_masked": round(float(mask.mean()), 4),
            "true_matches": n_true,
            "true_matches_masked": n_true_masked,
            "true_masked_rate": round(n_true_masked / max(n_true, 1), 5),
            "queries_all_true_masked": n_dead,
            "n_query": len(query)}


def evaluate_st(query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                embeddings: dict[str, np.ndarray], index: TrackletIndex,
                topo: CameraTopology, ranks=(1, 5, 10),
                overlap_thr: float = 0.02, margin: int = 0) -> dict:
    """Cosine retrieval with the ST impossibility mask applied to the distance
    matrix (masked pairs pushed to +inf before ranking)."""
    q, g = list(query), list(gallery)
    Q, G = _stack(q, embeddings), _stack(g, embeddings)
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - Q @ G.T
    dist[build_st_mask(q, g, index, topo, overlap_thr, margin)] = INF
    return _score(q, g, dist, ranks)


def evaluate_rerank_st(query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                       embeddings: dict[str, np.ndarray], index: TrackletIndex,
                       topo: CameraTopology, ranks=(1, 5, 10),
                       k1: int = 20, k2: int = 6,
                       overlap_thr: float = 0.02, margin: int = 0) -> dict:
    """CA-Jaccard re-ranked retrieval + ST mask (mask applied after re-ranking;
    the constraint is absolute so it overrides any similarity evidence)."""
    from .cajaccard import ca_jaccard_distance

    q, g = list(query), list(gallery)
    feats = np.concatenate([_stack(q, embeddings), _stack(g, embeddings)], axis=0)
    cams = [it.camera for it in q] + [it.camera for it in g]
    D = ca_jaccard_distance(feats, cams, k1=k1, k2=k2, camera_aware=True)
    dist = D[: len(q), len(q):].copy()
    dist[build_st_mask(q, g, index, topo, overlap_thr, margin)] = INF
    return _score(q, g, dist, ranks)
