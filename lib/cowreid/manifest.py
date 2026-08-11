"""Dataset manifest: parse the ``2025Sep18/<camera>/<gt_label>/<ts>.jpeg`` layout
into typed :class:`Sample` records and provide filtering / stats.

Ground-truth labels (``gt_label``) are individual cow identities. They are loaded
here for *evaluation* and for the scrambling control only. The self-supervised
pair miner must not consume ``gt_label`` directly (the one principled exception is
using it as a stand-in for a single-camera tracker when forming intra-camera
tracklets -- see :mod:`cowreid.tracklets`).
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


_TS_FMT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class Sample:
    """One crop (a single detection at one timestamp)."""

    path: str        # dataset-relative, forward-slashed, e.g. "2025Sep18/66.130/053/20250918_131207.jpeg"
    camera: str      # "66.130"
    gt_label: str    # "053"  -- identity; eval/scramble only
    t: int           # integer seconds (monotonic ordering key)
    ts_str: str      # "20250918_131207"

    @property
    def tracklet_group(self) -> tuple[str, str]:
        """(camera, gt_label) -- the proxy key a single-camera tracker would yield."""
        return (self.camera, self.gt_label)


def parse_path(path: str) -> Sample | None:
    """Parse a dataset-relative path into a :class:`Sample` (None if not a crop)."""
    norm = path.replace("\\", "/").strip().rstrip("/")
    if not norm.lower().endswith((".jpeg", ".jpg", ".png")):
        return None
    parts = norm.split("/")
    if len(parts) < 4:
        return None
    camera, gt_label, fname = parts[-3], parts[-2], parts[-1]
    ts_str = os.path.splitext(fname)[0]
    try:
        dt = datetime.strptime(ts_str, _TS_FMT)
    except ValueError:
        return None
    t = int(dt.replace(tzinfo=timezone.utc).timestamp())
    return Sample(path=norm, camera=camera, gt_label=gt_label, t=t, ts_str=ts_str)


class Manifest:
    """An immutable-ish collection of :class:`Sample` with convenience accessors."""

    def __init__(self, samples: Sequence[Sample]):
        self.samples: list[Sample] = list(samples)

    # ----- constructors ---------------------------------------------------- #
    @classmethod
    def from_paths(cls, paths: Iterable[str]) -> "Manifest":
        out = []
        for p in paths:
            s = parse_path(p)
            if s is not None:
                out.append(s)
        return cls(out)

    @classmethod
    def from_listing_file(cls, listing_path: str | os.PathLike) -> "Manifest":
        """Build from a text file of archive member names (one per line).

        Generate such a file cheaply with ``tar -tzf archive.tar.gz > listing.txt``.
        """
        with open(listing_path, "r", encoding="utf-8") as fh:
            return cls.from_paths(fh)

    @classmethod
    def from_dir(cls, root: str | os.PathLike) -> "Manifest":
        root = Path(root)
        paths = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                paths.append(rel.replace("\\", "/"))
        return cls.from_paths(paths)

    @classmethod
    def from_tar(cls, tar_path: str | os.PathLike) -> "Manifest":
        """Build directly from the archive (streams the index; slower than a listing)."""
        import tarfile

        with tarfile.open(tar_path, "r:*") as tf:
            return cls.from_paths(tf.getnames())

    # ----- accessors ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def cameras(self) -> list[str]:
        return sorted({s.camera for s in self.samples})

    def identities(self) -> list[str]:
        return sorted({s.gt_label for s in self.samples})

    def time_range(self) -> tuple[int, int]:
        ts = [s.t for s in self.samples]
        return (min(ts), max(ts)) if ts else (0, 0)

    def filter(self, cameras: Iterable[str] | None = None,
               identities: Iterable[str] | None = None,
               exclude_cameras: Iterable[str] | None = None) -> "Manifest":
        cams = set(cameras) if cameras is not None else None
        ids = set(identities) if identities is not None else None
        ex = set(exclude_cameras) if exclude_cameras is not None else set()
        out = [s for s in self.samples
               if (cams is None or s.camera in cams)
               and (ids is None or s.gt_label in ids)
               and s.camera not in ex]
        return Manifest(out)

    def by_timestamp(self) -> dict[int, dict[str, list[Sample]]]:
        """{t -> {camera -> [samples]}} -- the structure Tier-2 mining iterates over."""
        out: dict[int, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
        for s in self.samples:
            out[s.t][s.camera].append(s)
        return {t: dict(cam_map) for t, cam_map in out.items()}

    def identity_camera_counts(self) -> dict[str, set[str]]:
        """{identity -> set(cameras it appears in)}."""
        out: dict[str, set[str]] = defaultdict(set)
        for s in self.samples:
            out[s.gt_label].add(s.camera)
        return dict(out)

    def stats(self) -> dict:
        per_cam = Counter(s.camera for s in self.samples)
        id_cams = self.identity_camera_counts()
        ncam_hist = Counter(len(c) for c in id_cams.values())
        t0, t1 = self.time_range()
        return {
            "n_samples": len(self.samples),
            "n_cameras": len(per_cam),
            "n_identities": len(id_cams),
            "per_camera_counts": dict(sorted(per_cam.items())),
            "identities_by_n_cameras": dict(sorted(ncam_hist.items())),
            "time_span_seconds": t1 - t0,
        }

    def to_csv(self, path: str | os.PathLike):
        from .io_utils import write_csv

        return write_csv(
            path,
            ({"path": s.path, "camera": s.camera, "gt_label": s.gt_label,
              "t": s.t, "ts_str": s.ts_str} for s in self.samples),
            header=["path", "camera", "gt_label", "t", "ts_str"],
        )
