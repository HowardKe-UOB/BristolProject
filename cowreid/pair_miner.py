"""Temporal-Sync Pair Miner.

Turns a time-synchronised multi-camera crop collection into self-supervised
training signal, using ground-truth identities only at evaluation time.

Tier 1 -- within-camera tracklet positives
    Consecutive (or same-tracklet) frames are the same cow for free. Clean, dense,
    single-view. Bootstraps an appearance encoder.

Tier 2 -- cross-view temporal bag matching
    At each instant t, the bag of crops in camera A and the bag in camera B share an
    unknown correspondence. Slots where each camera shows exactly one cow are
    *unambiguous anchors* (free, correct). Ambiguous bags are matched with entropic
    OT (Sinkhorn) + mutual-NN, calibrated by the anchor distance distribution and
    refined over a few iterations with per-camera feature centering. These yield the
    HARD cross-view positives (e.g. dorsal view <-> oblique 66.130) that appearance
    NN alone would never find.

Tier 3 -- session-level cross-camera identity propagation
    Accepted Tier-2 crop matches vote on edges between tracklets. A constrained
    union-find (cannot-link: same camera + overlapping time => different cows)
    produces cross-camera pseudo-identities -- the label-free analogue of GT IDs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np

from .features import CachedFeatureStore
from .manifest import Manifest
from .sinkhorn import match_with_dustbin, mutual_matches, sinkhorn
from .topology import CameraTopology
from .tracklets import Tracklet, TrackletIndex


@dataclass
class Pair:
    a: str                       # crop path
    b: str                       # crop path
    tier: int
    weight: float                # confidence in (0, 1]
    source: str                  # "tracklet" | "anchor" | "ot"
    meta: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        r = {"tier": self.tier, "path_a": self.a, "path_b": self.b,
             "weight": round(self.weight, 4), "source": self.source}
        r.update({f"meta_{k}": v for k, v in self.meta.items()})
        return r


@dataclass
class TrackletCluster:
    pseudo_id: int
    tracklet_ids: list[str]


class TemporalSyncPairMiner:
    def __init__(self, manifest: Manifest, tracklets: list[Tracklet],
                 feature_store: CachedFeatureStore | None = None,
                 sinkhorn_eps: float = 0.1):
        self.manifest = manifest
        self.tracklets = tracklets
        self.index = TrackletIndex(tracklets)
        self.fs = feature_store
        self.eps = sinkhorn_eps

    # ------------------------------------------------------------------ #
    # Tier 1
    # ------------------------------------------------------------------ #
    def mine_tier1(self, mode: str = "adjacent", window_s: int = 2,
                   max_pairs_per_tracklet: int | None = None,
                   rng_seed: int = 0) -> list[Pair]:
        """Within-camera positives.

        mode="adjacent": consecutive frames within ``window_s`` seconds.
        mode="intra":    random pairs drawn from the same tracklet (pose variety).
        """
        rng = np.random.default_rng(rng_seed)
        pairs: list[Pair] = []
        for tr in self.tracklets:
            s = tr.samples
            if len(s) < 2:
                continue
            local: list[tuple[int, int]] = []
            if mode == "adjacent":
                for i in range(len(s) - 1):
                    if s[i + 1].t - s[i].t <= window_s:
                        local.append((i, i + 1))
            elif mode == "intra":
                cap = max_pairs_per_tracklet or len(s)
                for _ in range(cap):
                    i, j = rng.choice(len(s), size=2, replace=False)
                    local.append((int(i), int(j)))
            else:
                raise ValueError(f"unknown mode {mode!r}")
            if max_pairs_per_tracklet and len(local) > max_pairs_per_tracklet:
                idx = rng.choice(len(local), size=max_pairs_per_tracklet, replace=False)
                local = [local[k] for k in idx]
            for i, j in local:
                pairs.append(Pair(a=s[i].path, b=s[j].path, tier=1, weight=1.0,
                                  source="tracklet",
                                  meta={"camera": tr.camera, "tracklet": tr.tracklet_id,
                                        "dt": abs(s[j].t - s[i].t)}))
        return pairs

    # ------------------------------------------------------------------ #
    # Tier 2
    # ------------------------------------------------------------------ #
    def _cooccurrence_bags(self):
        """Yield (t, camA, camB, samplesA, samplesB) for co-observed camera pairs."""
        for t, cam_map in self.manifest.by_timestamp().items():
            present = [c for c in cam_map if cam_map[c]]
            for camA, camB in combinations(sorted(present), 2):
                yield t, camA, camB, cam_map[camA], cam_map[camB]

    def _allowed_pairs(self, topology, threshold, want_overlap):
        """Set of frozenset({camA,camB}) to mine, gated by topology."""
        present = set()
        for _t, cam_map in self.manifest.by_timestamp().items():
            cams = sorted(c for c in cam_map if cam_map[c])
            for i in range(len(cams)):
                for j in range(i + 1, len(cams)):
                    present.add(frozenset((cams[i], cams[j])))
        if topology is None:
            return present  # caller is warned in mine_tier2
        keep = (topology.overlapping_pairs(threshold) if want_overlap
                else topology.non_overlapping_pairs(threshold))
        return present & keep

    def mine_tier2(self, topology: CameraTopology | None = None,
                   overlap_threshold: float = 0.02, min_confidence: float = 0.5,
                   n_refine_iters: int = 2, camera_center: bool = True,
                   dustbin_quantile: float = 0.5, verbose: bool = False) -> list[Pair]:
        """Cross-view POSITIVES via dustbin-OT, mined only on OVERLAPPING camera pairs.

        Requires ``feature_store``. Pass a :class:`CameraTopology` so positives are
        only sought where the same animal can co-occur; with ``topology=None`` all
        co-present pairs are mined (warned -- most pairs do not overlap here).

        The reject (dustbin) option lets a bag decline to match when the true
        partner is absent -- essential because the partner is missing ~90% of the
        time even on overlapping pairs. The discredited 1-vs-1 "anchor" seed is
        NOT used (empirically ~0.5% correct on this dataset).
        """
        if self.fs is None:
            raise ValueError("mine_tier2 needs a feature_store (cross-view positives "
                             "require appearance features; there is no reliable "
                             "label-free anchor on this dataset).")
        if topology is None and verbose:
            print("  tier2: WARNING no topology -> mining all co-present pairs; "
                  "expect low precision (most pairs do not overlap).")
        allowed = self._allowed_pairs(topology, overlap_threshold, want_overlap=True)

        bags = []  # (t, camA, camB, [paths A], [paths B])
        for t, camA, camB, sa, sb in self._cooccurrence_bags():
            if frozenset((camA, camB)) in allowed:
                bags.append((t, camA, camB, [s.path for s in sa], [s.path for s in sb]))
        if not bags:
            if verbose:
                print("  tier2: no bags on overlapping pairs.")
            return []

        self.fs.precompute(sorted({p for (_, _, _, A, B) in bags for p in A + B}),
                           verbose=verbose)
        return self._match_overlap_bags(bags, min_confidence, n_refine_iters,
                                         camera_center, dustbin_quantile, verbose)

    def _match_overlap_bags(self, bags, min_conf, n_iters, camera_center,
                            dustbin_quantile, verbose) -> list[Pair]:
        feats = self.fs.get
        cam_mean: dict[str, np.ndarray] = {}

        def center(vecs, cam):
            return vecs - cam_mean[cam] if (camera_center and cam in cam_mean) else vecs

        if camera_center:
            cam_to_paths: dict[str, list[str]] = defaultdict(list)
            for (_, ca, cb, A, B) in bags:
                cam_to_paths[ca].extend(A); cam_to_paths[cb].extend(B)
            for cam, ps in cam_to_paths.items():
                cam_mean[cam] = feats(sorted(set(ps))).mean(axis=0)

        accepted: list[Pair] = []
        for it in range(max(1, n_iters)):
            accepted = []
            high_conf_by_cam: dict[str, list[str]] = defaultdict(list)
            for (t, ca, cb, A, B) in bags:
                cost = _pairwise_euclidean(center(feats(A), ca), center(feats(B), cb))
                for i, j, conf in match_with_dustbin(cost, eps=self.eps,
                                                     dustbin_quantile=dustbin_quantile):
                    if conf < min_conf:
                        continue
                    accepted.append(Pair(a=A[i], b=B[j], tier=2, weight=float(conf),
                                         source="ot",
                                         meta={"t": t, "camA": ca, "camB": cb,
                                               "dist": round(float(cost[i, j]), 4)}))
                    high_conf_by_cam[ca].append(A[i]); high_conf_by_cam[cb].append(B[j])
            if camera_center and it + 1 < n_iters:
                for cam, ps in high_conf_by_cam.items():
                    if ps:
                        cam_mean[cam] = feats(sorted(set(ps))).mean(axis=0)
            if verbose:
                print(f"    refine iter {it}: {len(accepted)} matches")
        return accepted

    def mine_tier2_negatives(self, topology: CameraTopology,
                             overlap_threshold: float = 0.02,
                             max_per_slot: int | None = 8, rng_seed: int = 0
                             ) -> list[Pair]:
        """Hard NEGATIVES (no features needed): crops co-occurring in NON-overlapping
        camera pairs are different animals (>98% reliable on this dataset)."""
        rng = np.random.default_rng(rng_seed)
        allowed = self._allowed_pairs(topology, overlap_threshold, want_overlap=False)
        negs: list[Pair] = []
        for t, camA, camB, sa, sb in self._cooccurrence_bags():
            if frozenset((camA, camB)) not in allowed:
                continue
            cand = [(x, y) for x in sa for y in sb]
            if max_per_slot and len(cand) > max_per_slot:
                cand = [cand[k] for k in rng.choice(len(cand), max_per_slot, replace=False)]
            for x, y in cand:
                negs.append(Pair(a=x.path, b=y.path, tier=2, weight=1.0, source="neg",
                                 meta={"t": t, "camA": camA, "camB": camB}))
        return negs

    # ------------------------------------------------------------------ #
    # Tier 3
    # ------------------------------------------------------------------ #
    def mine_tier3(self, tier2_pairs: Iterable[Pair], min_votes: int = 2,
                   min_weight: float = 1.0) -> tuple[list[TrackletCluster], dict]:
        """Propagate Tier-2 crop matches to cross-camera tracklet clusters.

        An edge between two tracklets accumulates the confidence of every crop match
        linking them; edges with >= ``min_votes`` matches and total weight >=
        ``min_weight`` are eligible for merging, subject to cannot-link constraints.
        """
        edge_w: dict[tuple[str, str], float] = defaultdict(float)
        edge_n: dict[tuple[str, str], int] = defaultdict(int)
        for p in tier2_pairs:
            ta, tb = self.index.tracklet_of(p.a), self.index.tracklet_of(p.b)
            if ta is None or tb is None or ta == tb:
                continue
            key = tuple(sorted((ta, tb)))
            edge_w[key] += p.weight
            edge_n[key] += 1

        cannot_link = self._cannot_link_pairs()
        uf = _ConstrainedUnionFind(cannot_link)
        for t in self.tracklets:
            uf.add(t.tracklet_id)

        for key in sorted(edge_w, key=lambda k: -edge_w[k]):
            if edge_n[key] < min_votes or edge_w[key] < min_weight:
                continue
            uf.union(*key)

        clusters: dict[int, list[str]] = defaultdict(list)
        for t in self.tracklets:
            clusters[uf.find(t.tracklet_id)].append(t.tracklet_id)
        out = [TrackletCluster(pseudo_id=i, tracklet_ids=sorted(tids))
               for i, tids in enumerate(clusters.values())]
        graph = {"edges": [{"a": k[0], "b": k[1], "weight": round(edge_w[k], 3),
                            "votes": edge_n[k]} for k in edge_w]}
        return out, graph

    def _cannot_link_pairs(self) -> set[frozenset]:
        """Same-camera tracklets that overlap in time are different cows."""
        cl: set[frozenset] = set()
        by_cam: dict[str, list[Tracklet]] = defaultdict(list)
        for t in self.tracklets:
            by_cam[t.camera].append(t)
        for cam_tracks in by_cam.values():
            cam_tracks.sort(key=lambda tr: tr.t_start)
            for i in range(len(cam_tracks)):
                for j in range(i + 1, len(cam_tracks)):
                    if cam_tracks[j].t_start > cam_tracks[i].t_end:
                        break
                    if cam_tracks[i].overlaps_in_time(cam_tracks[j]):
                        cl.add(frozenset((cam_tracks[i].tracklet_id,
                                          cam_tracks[j].tracklet_id)))
        return cl

    # ------------------------------------------------------------------ #
    # Evaluation (uses GT -- diagnostics only)
    # ------------------------------------------------------------------ #
    def evaluate_clusters(self, clusters: list[TrackletCluster]) -> dict:
        labels_true, labels_pred = [], []
        for c in clusters:
            for tid in c.tracklet_ids:
                labels_true.append(self.index.gt_of(tid))
                labels_pred.append(c.pseudo_id)
        purity = _purity(labels_true, labels_pred)
        out = {"n_clusters": len(clusters), "n_tracklets": len(labels_true),
               "n_gt_identities": len(set(labels_true)), "purity": round(purity, 4)}
        try:  # optional richer metrics
            from sklearn.metrics import (adjusted_rand_score,
                                         normalized_mutual_info_score)
            out["nmi"] = round(float(normalized_mutual_info_score(labels_true, labels_pred)), 4)
            out["ari"] = round(float(adjusted_rand_score(labels_true, labels_pred)), 4)
        except Exception:
            pass
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    d2 = np.maximum(aa + bb - 2.0 * A @ B.T, 0.0)
    return np.sqrt(d2)


def _purity(true: list, pred: list) -> float:
    if not true:
        return 0.0
    clusters: dict = defaultdict(lambda: defaultdict(int))
    for tlab, plab in zip(true, pred):
        clusters[plab][tlab] += 1
    correct = sum(max(counts.values()) for counts in clusters.values())
    return correct / len(true)


class _ConstrainedUnionFind:
    def __init__(self, cannot_link: set[frozenset]):
        self.parent: dict[str, str] = {}
        self.members: dict[str, set[str]] = {}
        self.cannot_link = cannot_link

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x
            self.members[x] = {x}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _violates(self, ra: str, rb: str) -> bool:
        ma, mb = self.members[ra], self.members[rb]
        small, large = (ma, mb) if len(ma) <= len(mb) else (mb, ma)
        for x in small:
            for y in large:
                if frozenset((x, y)) in self.cannot_link:
                    return True
        return False

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self._violates(ra, rb):
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members[rb]
        del self.members[rb]
        return True
