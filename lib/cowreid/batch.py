"""Turn Phase-1 signals into training batches.

Sampling is at the *tracklet* level (PK-style: P pseudo-ids x K tracklets) so that
positives (same pseudo-id) and topology cannot-link hard-negatives co-occur densely
in a batch. Each tracklet contributes one clip (T contiguous frames). Masks are
derived from the pseudo-labels (positives) and the cannot-link set (hard negatives).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .io_utils import ImageLoader
from .tracklets import TrackletIndex


@dataclass
class RelationBatch:
    tracklet_ids: list[str]
    clip_paths: list[list[str]]            # (B, T) relative paths
    cameras: list[str]
    pseudo_labels: torch.Tensor            # (B,)
    positive_mask: torch.Tensor            # (B, B) bool
    forbid_negative_mask: torch.Tensor     # (B, B) bool
    hard_negative_mask: torch.Tensor       # (B, B) bool
    cannot_link_pairs: torch.Tensor        # (M, 2) long


class RelationBatchBuilder:
    def __init__(self, index: TrackletIndex, pseudo_labels: dict[str, int],
                 cannot_link: set[frozenset], clip_len: int = 4, seed: int = 0):
        self.index = index
        self.labels = pseudo_labels
        self.cl = cannot_link
        self.clip_len = clip_len
        self.rng = np.random.default_rng(seed)
        self.by_label: dict[int, list[str]] = defaultdict(list)
        for tid, lab in pseudo_labels.items():
            self.by_label[lab].append(tid)

    def _sample_clip(self, tid: str) -> list[str]:
        paths = self.index[tid].paths
        T = self.clip_len
        if len(paths) <= T:
            return paths + [paths[-1]] * (T - len(paths))  # pad with last frame
        start = int(self.rng.integers(0, len(paths) - T + 1))
        return paths[start:start + T]

    def sample(self, num_pseudo_ids: int = 8, per_id: int = 2) -> RelationBatch:
        pool = [l for l in self.by_label if l >= 0]
        chosen = self.rng.choice(pool, size=min(num_pseudo_ids, len(pool)), replace=False)
        tids: list[str] = []
        for lab in chosen:
            cand = self.by_label[int(lab)]
            pick = self.rng.choice(cand, size=min(per_id, len(cand)),
                                   replace=len(cand) < per_id)
            tids.extend(pick.tolist())
        return self.build(tids)

    def build(self, tracklet_ids: Sequence[str]) -> RelationBatch:
        tids = list(tracklet_ids)
        B = len(tids)
        labs = torch.tensor([self.labels.get(t, -1) for t in tids], dtype=torch.long)

        pos = (labs.unsqueeze(0) == labs.unsqueeze(1)) & (labs.unsqueeze(0) >= 0)
        hard = torch.zeros(B, B, dtype=torch.bool)
        cl_pairs: list[tuple[int, int]] = []
        for i in range(B):
            for j in range(i + 1, B):
                if frozenset((tids[i], tids[j])) in self.cl:
                    hard[i, j] = hard[j, i] = True
                    cl_pairs.append((i, j))
        forbid = torch.zeros(B, B, dtype=torch.bool)  # extension point

        return RelationBatch(
            tracklet_ids=tids,
            clip_paths=[self._sample_clip(t) for t in tids],
            cameras=[self.index.camera_of(t) for t in tids],
            pseudo_labels=labs,
            positive_mask=pos,
            forbid_negative_mask=forbid,
            hard_negative_mask=hard,
            cannot_link_pairs=torch.tensor(cl_pairs, dtype=torch.long).reshape(-1, 2),
        )


def build_transform(image_size: int = 224):
    """Standard ImageNet-normalised eval transform (torchvision)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_clips(batch: RelationBatch, loader: ImageLoader, transform,
               device: str = "cpu") -> torch.Tensor:
    """Materialise clip pixels -> (B, T, C, H, W)."""
    clips = []
    for paths in batch.clip_paths:
        frames = [transform(loader.load(p)) for p in paths]
        clips.append(torch.stack(frames))
    return torch.stack(clips).to(device)
