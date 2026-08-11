"""Intra-Inter Camera Similarity (IICS, Xuan & Zhang, CVPR 2021), adapted to our
cached-feature cattle setting.

IICS decomposes cross-camera pseudo-labelling into two stages to beat the camera /
viewpoint domain gap S^c:
  * Intra-camera: cluster within each camera (low gap) -> per-camera pseudo-labels ->
    a multi-branch network (one cosine classifier per camera, shared embedding).
  * Inter-camera: represent each sample by its classification scores over ALL camera
    classifiers; the Jaccard similarity of these scores is robust to S^c, and is added
    to the feature cosine to cluster across cameras.

Adaptations for our data:
  * The "backbone" is a small head over FROZEN DINOv2 per-frame features (temporal
    pool + AIBN1d embedding) -- training is fast and reuses the feature cache.
  * Intra-camera labels exploit our tracklets (clean within-camera identity groups);
    same-camera time-overlap pairs are cannot-link.
  * The inter-camera clustering is additionally constrained by the topology
    cannot-link set (a cow cannot be in two places at once) -- our contribution.
  * AIBN is approximated for 1-D feature vectors (frozen backbone limits its role).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TemporalPool
from .pair_miner import _ConstrainedUnionFind


class AIBN1d(nn.Module):
    """Adaptive IN/BN for (B, D) feature vectors: linear fuse of batch-norm (per-dim)
    and instance standardisation (per-sample across dims), learnable mix alpha."""

    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        a = self.alpha.clamp(0.0, 1.0)
        bn = self.bn(x) if x.size(0) > 1 else x
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        instn = (x - mu) / torch.sqrt(var + 1e-5)
        return self.gamma * (a * bn + (1 - a) * instn) + self.beta


class MultiBranchReID(nn.Module):
    """Shared embedding over cached clip features + one cosine classifier per camera."""

    def __init__(self, in_dim: int, n_classes_per_cam: dict[str, int],
                 proj_dim: int = 256, pool: str = "attn"):
        super().__init__()
        self.pool = TemporalPool(in_dim, pool)
        self.embed = nn.Sequential(nn.Linear(in_dim, proj_dim), AIBN1d(proj_dim),
                                   nn.ReLU(inplace=True))
        self.proj_dim = proj_dim
        # nn.ModuleDict keys cannot contain "." -> sanitise camera ids
        self._key = {c: c.replace(".", "_") for c in n_classes_per_cam}
        self.classifiers = nn.ModuleDict(
            {self._key[c]: nn.Linear(proj_dim, max(1, k), bias=False)
             for c, k in n_classes_per_cam.items()})

    def backbone(self, clip):                      # (B, T, in_dim) -> (B, proj_dim) L2
        return F.normalize(self.embed(self.pool(clip)), dim=1)

    def logits(self, emb, camera, scale: float = 16.0):
        w = F.normalize(self.classifiers[self._key[camera]].weight, dim=1)
        return scale * F.linear(emb, w)            # cosine classifier

    def all_scores(self, emb, scale: float = 16.0):
        return torch.cat([F.softmax(scale * F.linear(emb, F.normalize(clf.weight, dim=1)), dim=1)
                          for clf in self.classifiers.values()], dim=1)   # (B, sum_c k_c)


# --------------------------------------------------------------------------- #
# inter-camera similarity & clustering
# --------------------------------------------------------------------------- #
def jaccard_from_scores(scores: np.ndarray, n_cameras: int) -> np.ndarray:
    """Pairwise Jaccard of concatenated softmax scores. Since each of the C softmaxes
    sums to 1, sum(s)=C, so Jaccard = SM / (2C - SM) with SM = sum_k min(s_m, s_n)."""
    N = scores.shape[0]
    SM = np.empty((N, N), dtype=np.float32)
    for i in range(N):                              # row-chunked min-sum
        SM[i] = np.minimum(scores[i][None, :], scores).sum(axis=1)
    return SM / (2.0 * n_cameras - SM + 1e-12)


def cluster_from_similarity(ids, sim: np.ndarray, threshold: float, k: int,
                            cannot_link: set[frozenset] | None) -> dict[str, int]:
    """Mutual-kNN + constrained union-find on a precomputed similarity matrix."""
    n = len(ids)
    s = sim.copy()
    np.fill_diagonal(s, -1e9)
    knn = np.argsort(-s, axis=1)[:, :k]
    edges = []
    for i in range(n):
        for j in knn[i]:
            if s[i, j] >= threshold and i in knn[j]:
                edges.append((s[i, j], i, int(j)))
    edges.sort(reverse=True)
    uf = _ConstrainedUnionFind(cannot_link or set())
    for t in ids:
        uf.add(t)
    for _w, i, j in edges:
        uf.union(ids[i], ids[j])
    comp = defaultdict(list)
    for t in ids:
        comp[uf.find(t)].append(t)
    labels, nxt = {}, 0
    for members in comp.values():
        for t in members:
            labels[t] = nxt
        nxt += 1
    return labels


@torch.no_grad()
def inter_camera_labels(model: MultiBranchReID, clips, tids, device, n_cameras,
                        mu: float = 1.0, threshold: float = 0.5, k: int = 10,
                        cannot_link=None) -> dict[str, int]:
    """Embeddings + classifier scores -> inter-camera similarity -> clusters."""
    model.eval()
    embs, scores = [], []
    for t in tids:
        x = torch.tensor(clips[t][None], dtype=torch.float32, device=device)
        e = model.backbone(x)
        embs.append(e[0].cpu().numpy())
        scores.append(model.all_scores(e)[0].cpu().numpy())
    E = np.stack(embs); S = np.stack(scores)
    feat_sim = E @ E.T                              # cosine (E is normalised)
    jac = jaccard_from_scores(S, n_cameras)
    sim = feat_sim + mu * jac
    return cluster_from_similarity(list(tids), sim, threshold, k, cannot_link)
