"""Crop-level cross-view positive mining via dustbin-OT, aggregated over time.

For each overlapping camera pair, at every co-occurring timestamp we match the
single-frame crops in camera A against those in camera B with dustbin-OT (a reject
option, since the true partner is usually absent). Each accepted crop match votes
for a link between the two crops' tracklets. Votes are summed over all timestamps,
so a genuinely co-travelling cross-camera tracklet pair accrues many votes while
spurious matches stay low -> the ``min_votes`` filter yields many, denoised
tracklet-level must-links (vs the handful from one-shot tracklet mutual-NN).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .sinkhorn import match_with_dustbin


def crossview_crop_bags(manifest, topo, keep_cameras, index, overlap_threshold=0.02,
                        max_bags=1500, seed=0):
    """Return (bags, crop_paths). bags = list of (camA, camB, [pathsA], [pathsB]) at
    co-occurring timestamps on overlapping camera pairs, restricted to ``keep_cameras``
    (and to crops that belong to an indexed tracklet). Subsampled to ``max_bags``."""
    ov = topo.overlapping_pairs(overlap_threshold)
    keep = set(keep_cameras)
    bags = []
    for _t, cam_map in manifest.by_timestamp().items():
        present = sorted(c for c in cam_map if c in keep and cam_map[c])
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                if frozenset((a, b)) not in ov:
                    continue
                A = [s.path for s in cam_map[a] if index.tracklet_of(s.path)]
                B = [s.path for s in cam_map[b] if index.tracklet_of(s.path)]
                if A and B:
                    bags.append((a, b, A, B))
    rng = np.random.default_rng(seed)
    if len(bags) > max_bags:
        bags = [bags[k] for k in rng.choice(len(bags), max_bags, replace=False)]
    crop_paths = sorted({p for _a, _b, A, B in bags for p in (A + B)})
    return bags, crop_paths


@torch.no_grad()
def embed_crops(encoder, image_loader, transform, paths, device, bs=64):
    """Embed single crops (as T=1 clips) -> {path: vector}."""
    encoder.eval()
    out = {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = torch.stack([transform(image_loader.load(p)) for p in chunk]).unsqueeze(1).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            e = encoder.embed(x)
        for p, v in zip(chunk, e.float().cpu().numpy()):
            out[p] = v
    return out


def mine_crop_ot_links(bags, crop_emb, path_to_tracklet, eps_ot=0.1, min_conf=0.5,
                       min_votes=3, gt=None):
    """Aggregate dustbin-OT crop matches into tracklet must-links.
    Returns (links, gt_precision, n_candidate_pairs)."""
    votes = defaultdict(float)
    counts = defaultdict(int)
    for _camA, _camB, A, B in bags:
        EA = np.stack([crop_emb[p] for p in A])
        EB = np.stack([crop_emb[p] for p in B])
        EA = EA / (np.linalg.norm(EA, axis=1, keepdims=True) + 1e-12)
        EB = EB / (np.linalg.norm(EB, axis=1, keepdims=True) + 1e-12)
        cost = 1.0 - EA @ EB.T
        for i, j, conf in match_with_dustbin(cost, eps=eps_ot):
            if conf < min_conf:
                continue
            ta, tb = path_to_tracklet(A[i]), path_to_tracklet(B[j])
            if ta and tb and ta != tb:
                key = frozenset((ta, tb))
                votes[key] += conf
                counts[key] += 1
    links = [k for k in votes if counts[k] >= min_votes]
    prec = None
    if gt is not None and links:
        prec = float(np.mean([gt[tuple(k)[0]] == gt[tuple(k)[1]] for k in links]))
    return links, prec, len(votes)
