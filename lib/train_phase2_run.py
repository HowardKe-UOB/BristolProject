"""Small Phase-2 training run: does the self-supervised objective lift cross-camera
Re-ID above the FROZEN DINOv2 baseline?

Strategy (fast + honest): freeze the DINOv2 backbone and cache its per-frame features
once, then train only a lightweight head (temporal-attention pool + projection MLP)
with the multi-task SSL objective (tracklet positives + Cluster-Contrast +
topology hard-negatives/cannot-link). No identity labels are used in training.

Two protocols:
  * full  -- transductive USL: train head on all multi-camera tracklets, eval all-vs-all.
  * loco  -- inductive cross-view: train head WITHOUT 66.130, eval query=66.130.

    python train_phase2_run.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cowreid import (CameraTopology, DinoV2Extractor, ImageLoader, Manifest,
                     build_tracklets, extract_paths)
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.encoder import TemporalPool
from cowreid.eval import EvalItem, ReIDEvaluator, build_full_cross_camera
from cowreid.features import CachedFeatureStore
from cowreid.tracklets import TrackletIndex
from train_phase2 import build_objective


# --------------------------------------------------------------------------- #
class FeatureClipEncoder(nn.Module):
    """Head over cached per-frame features: (B, T, in_dim) -> {"feat","embed"}."""

    def __init__(self, in_dim: int = 384, proj_dim: int = 256, pool: str = "attn"):
        super().__init__()
        self.pool = TemporalPool(in_dim, pool)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim))

    def forward(self, clip_feats):
        feat = self.pool(clip_feats)
        embed = F.normalize(self.proj(feat), dim=1)
        return {"feat": F.normalize(feat, dim=1), "embed": embed}


def sample_frames(tracklet, k):
    paths = tracklet.paths
    idx = np.linspace(0, len(paths) - 1, min(k, len(paths))).astype(int)
    return [paths[i] for i in idx]


def cache_features(tracklets, tar, work, cache_path, frames, device):
    import os
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return {k: v for k, v in zip(d["ids"], d["clips"])}
    paths_per = {t.tracklet_id: sample_frames(t, frames) for t in tracklets}
    allpaths = sorted({p for ps in paths_per.values() for p in ps})
    print(f"[cache] extracting {len(allpaths)} crops + DINOv2 features on {device} ...")
    extract_paths(tar, allpaths, work)
    fs = CachedFeatureStore(ImageLoader(root=work, tar_path=tar),
                            DinoV2Extractor(device=device))
    fs.precompute(allpaths, verbose=True)
    clips = {tid: np.stack(fs.get(ps)) for tid, ps in paths_per.items()}
    np.savez_compressed(cache_path, ids=np.array(list(clips), dtype=object),
                        clips=np.array(list(clips.values()), dtype=object))
    return clips


# --------------------------------------------------------------------------- #
def make_masks(tids, labels, cl):
    B = len(tids)
    labs = torch.tensor([labels.get(t, -1) for t in tids], dtype=torch.long)
    pos = (labs[:, None] == labs[None, :]) & (labs[None, :] >= 0)
    hard = torch.zeros(B, B, dtype=torch.bool)
    pairs = []
    for i in range(B):
        for j in range(i + 1, B):
            if frozenset((tids[i], tids[j])) in cl:
                hard[i, j] = hard[j, i] = True
                pairs.append((i, j))
    return labs, pos, hard, torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def frozen_embeddings(clips, tids):
    return {t: _norm(clips[t].mean(0)) for t in tids}


def head_embeddings(encoder, clips, tids, device):
    encoder.eval()
    out = {}
    with torch.no_grad():
        for t in tids:
            x = torch.tensor(clips[t][None], dtype=torch.float32, device=device)
            out[t] = encoder(x)["embed"][0].cpu().numpy()
    return out


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-12)


def train_head(clips, train_tids, cl, device, steps, refresh_every, P, K, T,
               proj_dim, seed, fixed_labels=None):
    """Train the head with clustered pseudo-labels, or with ``fixed_labels`` (e.g.
    GT) for the supervised upper bound -- in which case clustering/refresh is off."""
    rng = np.random.default_rng(seed)
    enc = FeatureClipEncoder(in_dim=clips[train_tids[0]].shape[1], proj_dim=proj_dim).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=3e-4, weight_decay=1e-4)

    def cluster_with(emb_dict):
        feats = np.stack([emb_dict[t] for t in train_tids])
        return ClusterAssigner(sim_threshold=0.6, k=10).assign(train_tids, feats, cl)

    supervised = fixed_labels is not None
    labels = (dict(fixed_labels) if supervised
              else cluster_with(frozen_embeddings(clips, train_tids)))
    objective, memory = build_objective(dim=proj_dim,
                                        num_clusters=max(1, ClusterAssigner.num_clusters(labels)))
    objective.to(device)

    for step in range(steps):
        if not supervised and step > 0 and step % refresh_every == 0:
            labels = cluster_with(head_embeddings(enc, clips, train_tids, device))
            memory.reset(max(1, ClusterAssigner.num_clusters(labels)))
        by_label = defaultdict(list)
        for t, l in labels.items():
            if l >= 0:
                by_label[l].append(t)
        if not by_label:
            break
        chosen = rng.choice(list(by_label), size=min(P, len(by_label)), replace=False)
        tids = []
        for l in chosen:
            cand = by_label[int(l)]
            tids += rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K).tolist()

        clip_batch = np.stack([clips[t][rng.integers(0, len(clips[t]), size=T)] for t in tids])
        x = torch.tensor(clip_batch, dtype=torch.float32, device=device)
        labs, pos, hard, cl_pairs = make_masks(tids, labels, cl)

        enc.train()
        embed = enc(x)["embed"]
        total, comp = objective(embed, positive_mask=pos.to(device),
                                hard_negative_mask=hard.to(device),
                                cluster_labels=labs.to(device),
                                cannot_link_pairs=cl_pairs.to(device))
        opt.zero_grad(); total.backward(); opt.step()
        if step % 100 == 0:
            print(f"  step {step:4d}  total={float(comp['total']):.3f}  "
                  f"con={float(comp['contrastive']):.3f}  "
                  f"clu={float(comp.get('cluster', torch.tensor(0))):.3f}  "
                  f"#clusters={ClusterAssigner.num_clusters(labels)}")
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--refresh-every", type=int, default=200)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)

    clips = cache_features(tracklets, args.tar, args.work, args.cache, args.frames, device)
    evaluator = ReIDEvaluator(ranks=(1, 5, 10))

    # ---- protocol FULL (transductive) ---- #
    full_q, full_g = build_full_cross_camera(tracklets)
    full_tids = sorted({it.tracklet_id for it in full_q})
    print("\n[FULL] frozen baseline:")
    fr = evaluator.evaluate(full_q, full_g, frozen_embeddings(clips, full_tids))
    print("   ", fr)
    print("[FULL] training head (transductive, no labels)...")
    enc_full = train_head(clips, full_tids, cl, device, args.steps, args.refresh_every,
                          args.P, args.K, args.T, args.proj_dim, args.seed)
    tr = evaluator.evaluate(full_q, full_g, head_embeddings(enc_full, clips, full_tids, device))
    print("[FULL] trained head:", tr)

    # ---- protocol LEAVE-OUT 66.130 (inductive cross-view) ---- #
    hold = args.holdout_camera
    train_tids = [t.tracklet_id for t in tracklets if t.camera != hold]
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
               for t in tracklets if t.camera != hold]
    gallery_ids = {t.gt_label for t in tracklets if t.camera != hold}
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gallery_ids]
    eval_tids = sorted({it.tracklet_id for it in query + gallery})
    print(f"\n[LOCO {hold}] frozen baseline:")
    fr2 = evaluator.evaluate(query, gallery, frozen_embeddings(clips, eval_tids))
    print("   ", fr2)
    print(f"[LOCO {hold}] training head WITHOUT {hold} (inductive)...")
    enc_loco = train_head(clips, train_tids, cl, device, args.steps, args.refresh_every,
                          args.P, args.K, args.T, args.proj_dim, args.seed)
    tr2 = evaluator.evaluate(query, gallery, head_embeddings(enc_loco, clips, eval_tids, device))
    print(f"[LOCO {hold}] trained head:", tr2)


if __name__ == "__main__":
    main()
