"""Evaluation splits and the label-scrambling control.

:class:`SplitGenerator`
    * tracklet-disjoint train/val/test (no frame leaks across splits),
    * identity-disjoint *open-set* ReID by default (test cows unseen in train),
    * cross-camera query/gallery construction with the standard same-camera-same-id
      "junk" rule,
    * a leave-one-camera-out protocol for the hard viewpoint (66.130).

:class:`LabelScrambler`
    * ``permute``   -- shuffle the label vector over tracklets (Zhang et al. style
      randomization / memorization control: appearance-label correspondence broken,
      class frequencies preserved),
    * ``symmetric`` -- corrupt a fraction ``noise_rate`` of labels to random other
      identities (annotation-noise robustness curve).
All operations are deterministic given ``seed``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .tracklets import Tracklet, TrackletIndex


@dataclass
class Split:
    name: str
    train: list[str] = field(default_factory=list)        # tracklet ids
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    query: list[dict] = field(default_factory=list)       # {tracklet_id, identity, camera}
    gallery: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class SplitGenerator:
    def __init__(self, tracklets: Sequence[Tracklet], seed: int = 0):
        self.tracklets = list(tracklets)
        self.index = TrackletIndex(self.tracklets)
        self.seed = seed
        self._id_to_cams: dict[str, set[str]] = defaultdict(set)
        self._id_to_tracks: dict[str, list[str]] = defaultdict(list)
        for t in self.tracklets:
            self._id_to_cams[t.gt_label].add(t.camera)
            self._id_to_tracks[t.gt_label].append(t.tracklet_id)

    # ------------------------------------------------------------------ #
    def make_split(self, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
                   disjoint_by: str = "identity", name: str = "random") -> Split:
        rng = np.random.default_rng(self.seed)
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

        if disjoint_by == "identity":
            multi = [i for i, c in self._id_to_cams.items() if len(c) >= 2]
            single = [i for i, c in self._id_to_cams.items() if len(c) < 2]
            rng.shuffle(multi)
            n_ids = len(self._id_to_cams)
            n_test, n_val = int(round(ratios[2] * n_ids)), int(round(ratios[1] * n_ids))
            # draw cross-camera-capable identities into test/val first
            test_ids = multi[:n_test]
            val_ids = multi[n_test:n_test + n_val]
            train_ids = multi[n_test + n_val:] + single
            sel = {"train": train_ids, "val": val_ids, "test": test_ids}
            split = Split(name=name, meta={"disjoint_by": "identity", "ratios": ratios})
            for part, ids in sel.items():
                tids = [tid for i in ids for tid in self._id_to_tracks[i]]
                getattr(split, part).extend(sorted(tids))
        elif disjoint_by == "tracklet":
            tids = [t.tracklet_id for t in self.tracklets]
            rng.shuffle(tids)
            n = len(tids)
            n_tr, n_va = int(round(ratios[0] * n)), int(round(ratios[1] * n))
            split = Split(name=name,
                          train=sorted(tids[:n_tr]),
                          val=sorted(tids[n_tr:n_tr + n_va]),
                          test=sorted(tids[n_tr + n_va:]),
                          meta={"disjoint_by": "tracklet", "ratios": ratios,
                                "warning": "identities may span splits (transductive)"})
        else:
            raise ValueError(f"unknown disjoint_by {disjoint_by!r}")

        split.query, split.gallery = self._query_gallery(split.test, rng)
        return split

    def make_leave_camera_out(self, holdout: str = "66.130",
                              val_ratio: float = 0.1) -> Split:
        """Cross-view domain generalisation: train on all other cameras, query the
        held-out camera against a gallery from the training cameras."""
        rng = np.random.default_rng(self.seed)
        train_all = [t for t in self.tracklets if t.camera != holdout]
        holdout_tracks = [t for t in self.tracklets if t.camera == holdout]
        gallery_ids = {t.gt_label for t in train_all}

        # carve a small val from training identities
        train_ids = sorted({t.gt_label for t in train_all})
        rng.shuffle(train_ids)
        n_val = int(round(val_ratio * len(train_ids)))
        val_ids = set(train_ids[:n_val])

        split = Split(name=f"leave_out_{holdout}",
                      meta={"protocol": "leave-camera-out", "holdout_camera": holdout})
        for t in train_all:
            (split.val if t.gt_label in val_ids else split.train).append(t.tracklet_id)
        split.test = [t.tracklet_id for t in holdout_tracks]
        split.train.sort(); split.val.sort(); split.test.sort()

        # query = held-out-camera tracklets whose identity exists in the gallery
        split.query = [{"tracklet_id": t.tracklet_id, "identity": t.gt_label,
                        "camera": t.camera}
                       for t in holdout_tracks if t.gt_label in gallery_ids]
        split.gallery = [{"tracklet_id": t.tracklet_id, "identity": t.gt_label,
                          "camera": t.camera} for t in train_all]
        split.meta["n_query_identities"] = len({q["identity"] for q in split.query})
        return split

    def _query_gallery(self, test_tids: Sequence[str], rng) -> tuple[list[dict], list[dict]]:
        """Standard cross-camera protocol: gallery = all test tracklets; query = one
        tracklet per (identity, query-camera) for identities seen in >=2 cameras.
        Eval must ignore gallery items with the same camera AND identity as a query."""
        by_id_cam: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for tid in test_tids:
            tr = self.index[tid]
            by_id_cam[tr.gt_label][tr.camera].append(tid)

        gallery = [{"tracklet_id": tid, "identity": self.index.gt_of(tid),
                    "camera": self.index.camera_of(tid)} for tid in test_tids]
        query: list[dict] = []
        for ident, cam_map in by_id_cam.items():
            if len(cam_map) < 2:
                continue  # not cross-camera-evaluable
            qcam = sorted(cam_map)[int(rng.integers(len(cam_map)))]
            qtid = sorted(cam_map[qcam])[0]
            query.append({"tracklet_id": qtid, "identity": ident, "camera": qcam})
        return query, gallery


class LabelScrambler:
    """Build scrambled identity labels for control experiments.

    Returns a ``{unit_id -> scrambled_label}`` mapping where ``unit_id`` is a
    tracklet id (level='tracklet', recommended) or a crop path (level='frame').
    """

    def __init__(self, tracklets: Sequence[Tracklet], seed: int = 0):
        self.tracklets = list(tracklets)
        self.seed = seed

    def _units(self, level: str) -> list[tuple[str, str]]:
        if level == "tracklet":
            return [(t.tracklet_id, t.gt_label) for t in self.tracklets]
        if level == "frame":
            return [(s.path, s.gt_label) for t in self.tracklets for s in t.samples]
        raise ValueError(f"unknown level {level!r}")

    def scramble(self, level: str = "tracklet", mode: str = "permute",
                 noise_rate: float = 1.0) -> dict[str, str]:
        rng = np.random.default_rng(self.seed)
        units = self._units(level)
        keys = [k for k, _ in units]
        labels = np.array([lab for _, lab in units], dtype=object)

        if mode == "permute":
            n = len(labels)
            k = int(round(noise_rate * n))
            idx = rng.permutation(n)[:k]
            shuffled = labels[idx].copy()
            rng.shuffle(shuffled)
            out = labels.copy()
            out[idx] = shuffled
        elif mode == "symmetric":
            classes = sorted(set(labels.tolist()))
            out = labels.copy()
            for i, lab in enumerate(labels):
                if rng.random() < noise_rate:
                    choice = lab
                    while choice == lab and len(classes) > 1:
                        choice = classes[int(rng.integers(len(classes)))]
                    out[i] = choice
        else:
            raise ValueError(f"unknown mode {mode!r}")

        return {k: str(v) for k, v in zip(keys, out.tolist())}

    @staticmethod
    def to_csv(mapping: dict[str, str], path: str):
        from .io_utils import write_csv

        return write_csv(path,
                         ({"unit": k, "scrambled_label": v} for k, v in mapping.items()),
                         header=["unit", "scrambled_label"])
