"""Phase-2 self-supervised losses (PyTorch).

Three components, matching the Phase-1 signals:

* :class:`NegativeAwareContrastiveLoss` -- multi-positive InfoNCE (SupCon-style) that
  (a) draws positives from Tier-1 tracklets and Tier-3 pseudo-clusters, (b) *excludes*
  forbidden pairs from the negative set (false-negative suppression), and (c) injects
  and up-weights the Tier-2 temporal *hard negatives*.
* :class:`ClusterMemory` + :class:`ClusterContrastLoss` -- Cluster-Contrast: a momentum
  memory of pseudo-identity centroids with a ClusterNCE objective; cross-camera
  positives come from the (periodically recomputed) cluster assignment.
* :class:`CannotLinkLoss` -- explicit topology cannot-link penalty pushing
  same-instant non-overlap-camera pairs apart (complements the contrastive negatives).

:class:`MultiTaskSSLObjective` weights and combines them. All forward signatures take
plain tensors/masks so the losses stay decoupled from the data layer.

These modules are not imported by ``cowreid/__init__`` so that the (torch-free)
Phase-1 tooling keeps importing without PyTorch. Use ``from cowreid.losses import ...``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeAwareContrastiveLoss(nn.Module):
    """Multi-positive InfoNCE with negative masking and hard-negative weighting.

    forward args (B = batch size of instances):
      z                    : (B, D) embeddings (L2-normalised internally).
      positive_mask        : (B, B) bool; True where i,j are positives.
      forbid_negative_mask : (B, B) bool; pairs that must NOT act as negatives
                             (e.g. same predicted cluster, or unknown relation).
      hard_negative_mask   : (B, B) bool; designated temporal hard negatives.
      hard_negative_weight : multiplier on hard negatives in the denominator.
    Anchors with no positive in the batch are ignored.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, z, positive_mask, forbid_negative_mask=None,
                hard_negative_mask=None, hard_negative_weight: float = 1.0):
        z = F.normalize(z, dim=1)
        B = z.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = (z @ z.t()) / self.t

        pos = positive_mask & ~eye
        forbid = (forbid_negative_mask if forbid_negative_mask is not None
                  else torch.zeros_like(eye))
        neg = ~pos & ~forbid & ~eye
        included = pos | neg

        weight = torch.ones_like(sim)
        if hard_negative_mask is not None and hard_negative_weight != 1.0:
            weight = torch.where(hard_negative_mask & neg,
                                 torch.full_like(sim, float(hard_negative_weight)),
                                 weight)

        # log-sum-exp denominator over the included set, stabilised
        masked = sim.masked_fill(~included, float("-inf"))
        row_max = masked.max(dim=1, keepdim=True).values
        row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
        exp_sim = torch.exp(sim - row_max) * weight * included
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = (sim - row_max) - torch.log(denom)

        pos_count = pos.sum(dim=1)
        loss_per_anchor = -(log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
        valid = pos_count > 0
        if valid.any():
            return loss_per_anchor[valid].mean()
        return (z.sum() * 0.0)  # no positives this batch -> zero, graph intact


class ClusterMemory(nn.Module):
    """Momentum memory bank of pseudo-identity centroids (Cluster-Contrast)."""

    def __init__(self, dim: int, num_clusters: int, temperature: float = 0.05,
                 momentum: float = 0.2):
        super().__init__()
        self.t = temperature
        self.m = momentum
        self.register_buffer("centroids", torch.zeros(num_clusters, dim))

    @property
    def num_clusters(self) -> int:
        return self.centroids.size(0)

    @torch.no_grad()
    def reset(self, num_clusters: int, dim: int | None = None):
        d = dim or self.centroids.size(1)
        self.centroids = torch.zeros(num_clusters, d, device=self.centroids.device)

    @torch.no_grad()
    def init_centroids(self, features, labels):
        feats = F.normalize(features, dim=1)
        for k in range(self.num_clusters):
            mask = labels == k
            if mask.any():
                self.centroids[k] = F.normalize(feats[mask].mean(0), dim=0)

    @torch.no_grad()
    def update(self, features, labels):
        feats = F.normalize(features, dim=1)
        for f, y in zip(feats, labels.tolist()):
            if y < 0:
                continue
            self.centroids[y] = F.normalize(
                self.m * self.centroids[y] + (1 - self.m) * f, dim=0)

    def forward(self, q, labels):
        """ClusterNCE: cross-entropy of q-to-centroid logits at the pseudo-label.
        Labels == -1 (clustering outliers) are ignored."""
        q = F.normalize(q, dim=1)
        # clone so a subsequent in-place momentum update cannot invalidate the
        # tensor autograd saved for this matmul's backward
        logits = (q @ self.centroids.clone().t()) / self.t
        return F.cross_entropy(logits, labels, ignore_index=-1)


class ClusterContrastLoss(nn.Module):
    """Thin wrapper: compute ClusterNCE and momentum-update the memory."""

    def __init__(self, memory: ClusterMemory):
        super().__init__()
        self.memory = memory

    def forward(self, embeddings, labels):
        loss = self.memory(embeddings, labels)
        self.memory.update(embeddings.detach(), labels)
        return loss


class CannotLinkLoss(nn.Module):
    """Hinge penalty on the cosine similarity of cannot-link pairs.

    cannot_link_pairs: LongTensor (M, 2) of batch indices. Penalises pairs whose
    cosine similarity exceeds ``margin`` (i.e. pushes them apart)."""

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, cannot_link_pairs):
        if cannot_link_pairs is None or cannot_link_pairs.numel() == 0:
            return z.sum() * 0.0
        z = F.normalize(z, dim=1)
        a = z[cannot_link_pairs[:, 0]]
        b = z[cannot_link_pairs[:, 1]]
        cos = (a * b).sum(dim=1)
        return F.relu(cos - self.margin).mean()


class MultiTaskSSLObjective(nn.Module):
    """Weighted sum of the three losses.

    forward returns ``(total_loss, components_dict)``. Pass ``cluster_labels=None`` to
    disable the clustering term for a batch (e.g. before the first cluster assignment).
    """

    def __init__(self, contrastive: NegativeAwareContrastiveLoss,
                 cluster: ClusterContrastLoss | None,
                 cannotlink: CannotLinkLoss,
                 w_contrastive: float = 1.0, w_cluster: float = 1.0,
                 w_cannotlink: float = 0.5, hard_negative_weight: float = 2.0):
        super().__init__()
        self.contrastive = contrastive
        self.cluster = cluster
        self.cannotlink = cannotlink
        self.w_con = w_contrastive
        self.w_clu = w_cluster
        self.w_cl = w_cannotlink
        self.hard_negative_weight = hard_negative_weight

    def forward(self, embeddings, positive_mask, hard_negative_mask=None,
                forbid_negative_mask=None, cluster_labels=None,
                cannot_link_pairs=None):
        comp = {}
        l_con = self.contrastive(embeddings, positive_mask, forbid_negative_mask,
                                 hard_negative_mask, self.hard_negative_weight)
        comp["contrastive"] = l_con.detach()
        total = self.w_con * l_con

        if self.cluster is not None and cluster_labels is not None:
            l_clu = self.cluster(embeddings, cluster_labels)
            comp["cluster"] = l_clu.detach()
            total = total + self.w_clu * l_clu

        l_cl = self.cannotlink(embeddings, cannot_link_pairs)
        comp["cannot_link"] = l_cl.detach()
        total = total + self.w_cl * l_cl

        comp["total"] = total.detach()
        return total, comp
