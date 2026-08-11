"""Tracklet construction.

A *tracklet* is a maximal temporally-contiguous run of crops of one object within
one camera. We build tracklets by grouping samples by ``(camera, gt_label)`` and
splitting wherever the time gap exceeds ``max_gap_s``.

IMPORTANT (honesty about supervision): grouping by ``gt_label`` here is a stand-in
for the output of a *single-camera multi-object tracker* (ByteTrack/SORT/...),
which in practice yields "these consecutive detections are the same object" for
free. It does NOT reveal the cross-camera identity -- that linkage is exactly what
the Tier-2/Tier-3 miner must recover without labels. This matches the standard
unsupervised tracklet-ReID assumption (e.g. TAUDL/UTAL). If you have real tracker
output, pass it via ``track_key_fn`` to remove the gt_label dependency entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .manifest import Manifest, Sample


@dataclass
class Tracklet:
    tracklet_id: str
    camera: str
    gt_label: str               # eval only
    samples: list[Sample] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        return [s.path for s in self.samples]

    @property
    def n_frames(self) -> int:
        return len(self.samples)

    @property
    def t_start(self) -> int:
        return self.samples[0].t

    @property
    def t_end(self) -> int:
        return self.samples[-1].t

    def overlaps_in_time(self, other: "Tracklet", slack: int = 0) -> bool:
        return (self.t_start - slack) <= other.t_end and (other.t_start - slack) <= self.t_end


def build_tracklets(manifest: Manifest, max_gap_s: int = 2,
                    track_key_fn: Callable[[Sample], tuple] | None = None) -> list[Tracklet]:
    """Split each (camera, identity) group into contiguous tracklets.

    Parameters
    ----------
    max_gap_s : a gap larger than this (in seconds) between consecutive frames
        starts a new tracklet. With ~1 fps capture, 2 tolerates a couple of dropped
        detections; set very large (e.g. 10**9) to treat each (camera, identity)
        group as a single tracklet.
    track_key_fn : optional override returning the grouping key for a sample. Use
        this to feed real tracker IDs instead of the gt_label proxy.
    """
    key_fn = track_key_fn or (lambda s: s.tracklet_group)
    groups: dict[tuple, list[Sample]] = {}
    for s in manifest:
        groups.setdefault(key_fn(s), []).append(s)

    tracklets: list[Tracklet] = []
    for key, samples in groups.items():
        samples.sort(key=lambda s: s.t)
        camera, gt_label = samples[0].camera, samples[0].gt_label
        run: list[Sample] = []
        idx = 0

        def flush(run_samples, run_idx):
            if run_samples:
                tracklets.append(Tracklet(
                    tracklet_id=f"{camera}/{gt_label}/{run_idx}",
                    camera=camera, gt_label=gt_label, samples=list(run_samples)))

        for s in samples:
            if run and (s.t - run[-1].t) > max_gap_s:
                flush(run, idx)
                idx += 1
                run = []
            run.append(s)
        flush(run, idx)
    tracklets.sort(key=lambda tr: (tr.camera, tr.gt_label, tr.t_start))
    return tracklets


class TrackletIndex:
    """Fast lookups over a tracklet collection."""

    def __init__(self, tracklets: Sequence[Tracklet]):
        self.tracklets = list(tracklets)
        self.by_id = {t.tracklet_id: t for t in self.tracklets}
        self._path_to_tid = {}
        for t in self.tracklets:
            for s in t.samples:
                self._path_to_tid[s.path] = t.tracklet_id

    def tracklet_of(self, path: str) -> str | None:
        return self._path_to_tid.get(path)

    def __getitem__(self, tid: str) -> Tracklet:
        return self.by_id[tid]

    def __len__(self) -> int:
        return len(self.tracklets)

    def gt_of(self, tid: str) -> str:
        return self.by_id[tid].gt_label

    def camera_of(self, tid: str) -> str:
        return self.by_id[tid].camera
