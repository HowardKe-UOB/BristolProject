"""Cross-camera Re-ID evaluation (Market-1501 protocol, tracklet-to-tracklet).

For each query tracklet, gallery tracklets are ranked by ascending embedding
distance. The standard junk rule removes gallery items with the **same identity
AND same camera** as the query (including the query's own tracklet), so a hit must
come from a *different camera* -- i.e. genuine cross-camera Re-ID. Reports CMC
(rank-k) and mean Average Precision (mAP). Queries with no cross-camera match in
the gallery are skipped.

GT identities are used here for scoring only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tracklets import Tracklet


@dataclass
class EvalItem:
    tracklet_id: str
    identity: str
    camera: str


def build_full_cross_camera(tracklets: Sequence[Tracklet]) -> tuple[list[EvalItem], list[EvalItem]]:
    """Diagnostic protocol: query = gallery = every tracklet whose identity appears
    in >= 2 cameras (the junk rule enforces cross-camera matching). Many queries ->
    a stable headline number."""
    id_cams: dict[str, set[str]] = defaultdict(set)
    for t in tracklets:
        id_cams[t.gt_label].add(t.camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.gt_label in multi]
    return items, list(items)


class ReIDEvaluator:
    def __init__(self, ranks: Sequence[int] = (1, 5, 10)):
        self.ranks = tuple(ranks)

    def evaluate(self, query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                 embeddings: dict[str, np.ndarray]) -> dict:
        """embeddings: {tracklet_id -> 1-D vector}. Returns mAP + CMC + counts."""
        q = list(query)
        g = list(gallery)
        Q = _stack(q, embeddings)
        G = _stack(g, embeddings)
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        G /= (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - Q @ G.T                                  # cosine distance (Nq, Ng)

        g_ids = np.array([it.identity for it in g])
        g_cams = np.array([it.camera for it in g])

        aps, cmc = [], {r: [] for r in self.ranks}
        for i, qi in enumerate(q):
            order = np.argsort(dist[i], kind="stable")
            ids_o, cams_o = g_ids[order], g_cams[order]
            keep = ~((ids_o == qi.identity) & (cams_o == qi.camera))  # drop junk
            matches = (ids_o[keep] == qi.identity)
            if not matches.any():
                continue                                       # no cross-cam match
            for r in self.ranks:
                cmc[r].append(bool(matches[:r].any()))
            cumsum = np.cumsum(matches)
            ap = (cumsum[matches] / (np.flatnonzero(matches) + 1)).mean()
            aps.append(ap)

        out = {"mAP": round(float(np.mean(aps)), 4) if aps else 0.0,
               "n_query_scored": len(aps), "n_query_total": len(q),
               "n_gallery": len(g)}
        for r in self.ranks:
            out[f"rank-{r}"] = round(float(np.mean(cmc[r])), 4) if cmc[r] else 0.0
        return out


def _score(query, gallery, dist, ranks):
    """CMC/mAP from a precomputed (Nq, Ng) distance matrix with the cross-camera junk rule."""
    g_ids = np.array([it.identity for it in gallery])
    g_cams = np.array([it.camera for it in gallery])
    aps, cmc = [], {r: [] for r in ranks}
    for i, qi in enumerate(query):
        order = np.argsort(dist[i], kind="stable")
        ids_o, cams_o = g_ids[order], g_cams[order]
        keep = ~((ids_o == qi.identity) & (cams_o == qi.camera))
        matches = ids_o[keep] == qi.identity
        if not matches.any():
            continue
        for r in ranks:
            cmc[r].append(bool(matches[:r].any()))
        cumsum = np.cumsum(matches)
        aps.append((cumsum[matches] / (np.flatnonzero(matches) + 1)).mean())
    out = {"mAP": round(float(np.mean(aps)), 4) if aps else 0.0,
           "n_query_scored": len(aps), "n_gallery": len(gallery)}
    for r in ranks:
        out[f"rank-{r}"] = round(float(np.mean(cmc[r])), 4) if cmc[r] else 0.0
    return out


def evaluate_rerank(query: Sequence[EvalItem], gallery: Sequence[EvalItem],
                    embeddings: dict[str, np.ndarray], ranks=(1, 5, 10),
                    k1: int = 20, k2: int = 6) -> dict:
    """k-reciprocal (camera-aware Jaccard) re-ranked retrieval. Computes the CA-Jaccard
    distance on the combined query+gallery set and scores query-vs-gallery."""
    from .cajaccard import ca_jaccard_distance

    q, g = list(query), list(gallery)
    feats = np.concatenate([_stack(q, embeddings), _stack(g, embeddings)], axis=0)
    cams = [it.camera for it in q] + [it.camera for it in g]
    D = ca_jaccard_distance(feats, cams, k1=k1, k2=k2, camera_aware=True)
    return _score(q, g, D[: len(q), len(q):], ranks)


def _stack(items: Sequence[EvalItem], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    missing = [it.tracklet_id for it in items if it.tracklet_id not in embeddings]
    if missing:
        raise KeyError(f"{len(missing)} tracklets missing embeddings, e.g. {missing[:3]}")
    return np.stack([embeddings[it.tracklet_id] for it in items]).astype(np.float64)
