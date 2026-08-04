"""I/O helpers: never-overwrite versioned paths, CSV/JSON writers, and a
tar image reader/extractor.

Design notes
------------
* Per project policy we *never* overwrite an existing output file. ``versioned_path``
  appends ``_v1``, ``_v2``, ... before the suffix until the name is free.
* Random access into a ``.tar.gz`` is O(n) (gzip is not cheaply seekable), so for
  feature extraction we extract the needed crops *once* into a working directory
  with :func:`extract_paths` and then read them with a plain :class:`ImageLoader`.
"""
from __future__ import annotations

import csv
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# --------------------------------------------------------------------------- #
# Versioned (never-overwrite) output paths
# --------------------------------------------------------------------------- #
def versioned_path(path: str | os.PathLike, start: int = 1) -> Path:
    """Return a path that does not yet exist.

    If ``foo.csv`` is free it is returned unchanged; otherwise ``foo_v1.csv``,
    ``foo_v2.csv`` ... is returned (first free one).
    """
    p = Path(path)
    if not p.exists():
        return p
    stem, suffix, parent = p.stem, p.suffix, p.parent
    i = start
    while True:
        cand = parent / f"{stem}_v{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def write_csv(path: str | os.PathLike, rows: Iterable[Mapping], header: Sequence[str]) -> Path:
    """Write ``rows`` (dicts) to a CSV at a versioned path. Returns the path used."""
    out = versioned_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out


def write_json(path: str | os.PathLike, obj) -> Path:
    out = versioned_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return out


# --------------------------------------------------------------------------- #
# Tar helpers
# --------------------------------------------------------------------------- #
def extract_paths(tar_path: str | os.PathLike, rel_paths: Iterable[str],
                  dest_dir: str | os.PathLike, skip_existing: bool = True) -> int:
    """Extract only ``rel_paths`` from a tarball into ``dest_dir``.

    Returns the number of files written. Reads the archive sequentially (one
    pass), which is the efficient access pattern for gzip.
    """
    wanted = set(rel_paths)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            name = member.name
            if name not in wanted:
                continue
            target = dest / name
            if skip_existing and target.exists():
                wanted.discard(name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                out.write(src.read())
            written += 1
            wanted.discard(name)
            if not wanted:
                break
    return written


class ImageLoader:
    """Loads crop pixels by relative path.

    Prefer ``root`` (an extracted directory) for speed. ``tar_path`` is supported
    as a fallback but re-scans the archive per call -- only use it for a few images.
    """

    def __init__(self, root: str | os.PathLike | None = None,
                 tar_path: str | os.PathLike | None = None):
        if root is None and tar_path is None:
            raise ValueError("ImageLoader needs either root= or tar_path=")
        self.root = Path(root) if root is not None else None
        self.tar_path = Path(tar_path) if tar_path is not None else None

    def load(self, rel_path: str):
        """Return a PIL.Image (RGB). Pillow is imported lazily."""
        from PIL import Image
        import io

        if self.root is not None:
            fp = self.root / rel_path
            if fp.exists():
                return Image.open(fp).convert("RGB")
            if self.tar_path is None:
                raise FileNotFoundError(fp)
        # tar fallback (slow)
        with tarfile.open(self.tar_path, "r:*") as tf:
            src = tf.extractfile(rel_path)
            if src is None:
                raise FileNotFoundError(rel_path)
            return Image.open(io.BytesIO(src.read())).convert("RGB")
