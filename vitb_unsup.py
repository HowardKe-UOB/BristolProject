"""ViT-B unsupervised, IO-fixed + checkpoint-resumable, to beat the environment.

Two problems solved:
  * IO bottleneck -> a one-time uint8 image cache on disk (_imgcache.npy, mmap'd),
    so training reads pre-resized 518x518 crops from RAM, not by decoding JPEGs.
  * Session teardown kills long runs -> the model checkpoints to disk every chunk;
    each foreground turn resumes, trains ~--wall seconds, saves, exits. Run it across
    a few turns until --target steps, then --eval.

Build cache once:   python vitb_unsup.py --build-cache
Train a chunk:      python vitb_unsup.py --wall 480 --target 1000
Final eval:         python vitb_unsup.py --wall 480 --target 1000 --eval
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from cowreid import CameraTopology, ImageLoader, Manifest, build_tracklets, extract_paths
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags, mine_crop_ot_links
from cowreid.eval import EvalItem, ReIDEvaluator, evaluate_rerank
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import FineTuneIICS, make_masks, merge_labels
from train_phase2 import build_objective
from train_phase2_run import sample_frames

VITB = "vit_base_patch14_dinov2.lvd142m"
CACHE_NPY = "_imgcache.npy"
CACHE_JSON = "_imgcache_paths.json"
CKPT = "_vitb_unsup_ckpt.pt"
HOLD = "66.130"
IMG = 518


def needed_paths(tracklets, index, topo, frames, max_bags):
    paths = {p for t in tracklets for p in sample_frames(t, frames)}
    train_cams = [c for c in {t.camera for t in tracklets} if c != HOLD]
    bags, crop_paths = crossview_crop_bags(
        Manifest([s for t in tracklets for s in t.samples]), topo, train_cams, index,
        max_bags=max_bags)
    paths.update(crop_paths)
    return sorted(paths), bags


def build_cache(args, tracklets, index, topo):
    paths, _bags = needed_paths(tracklets, index, topo, args.frames, args.max_bags)
    print(f"[build-cache] {len(paths)} crops -> resizing to {IMG}", flush=True)
    extract_paths(args.tar, paths, args.work)
    loader = ImageLoader(root=args.work)
    arr = np.zeros((len(paths), IMG, IMG, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        arr[i] = np.asarray(loader.load(p).resize((IMG, IMG)), dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            print(f"  cached {i + 1}/{len(paths)}", flush=True)
    np.save(CACHE_NPY, arr)
    json.dump({p: i for i, p in enumerate(paths)}, open(CACHE_JSON, "w"))
    print(f"[build-cache] saved {CACHE_NPY} ({arr.nbytes/1e9:.1f} GB)", flush=True)


class CacheLoader:
    """Fast clip/crop tensors from the mmap'd uint8 cache; normalise on GPU."""

    def __init__(self, cache, p2r, index, frames, device, rng):
        self.cache, self.p2r, self.index = cache, p2r, index
        self.paths = {t.tracklet_id: sample_frames(t, frames) for t in index.tracklets}
        self.device, self.rng = device, rng
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _norm(self, rows):
        u8 = torch.from_numpy(np.ascontiguousarray(self.cache[rows])).permute(0, 3, 1, 2)
        x = u8.to(self.device).float().div_(255.0)
        return (x - self.mean) / self.std                       # (N, 3, H, W)

    def _clip_rows(self, tid, T, train):
        ps = self.paths[tid]; n = len(ps)
        if train:
            idx = self.rng.integers(0, n, size=T)
        else:
            idx = np.linspace(0, n - 1, min(T, n)).astype(int)
            if len(idx) < T:
                idx = np.concatenate([idx, np.full(T - len(idx), idx[-1])])
        return [self.p2r[ps[i]] for i in idx]

    def batch(self, tids, T, train=True):
        rows = [r for t in tids for r in self._clip_rows(t, T, train)]
        return self._norm(rows).view(len(tids), T, 3, IMG, IMG)

    def crops(self, paths):
        return self._norm([self.p2r[p] for p in paths]).unsqueeze(1)


@torch.no_grad()
def embed_tids(model, cload, tids, T, device, bs=16):
    model.eval(); out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.embed(cload.batch(chunk, T, train=False))
        for t, v in zip(chunk, e.float().cpu().numpy()):
            out[t] = v
    return out


@torch.no_grad()
def embed_crops_cached(model, cload, paths, device, bs=64):
    model.eval(); out = {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.embed(cload.crops(chunk))
        for p, v in zip(chunk, e.float().cpu().numpy()):
            out[p] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--tar", default="2025Sep18.tar.gz")
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=480)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    t0 = time.time()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    if args.build_cache:
        build_cache(args, tracklets, index, topo)
        return

    # mmap cache + bags
    cache = np.load(CACHE_NPY, mmap_mode="r")
    p2r = json.load(open(CACHE_JSON))
    _, bags = needed_paths(tracklets, index, topo, args.frames, args.max_bags)
    crop_paths = sorted({p for _a, _b, A, B in bags for p in (A + B)})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(args.seed))

    d = np.load(args.vits_cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    loco_train = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    train_cams = [index.camera_of(t) for t in loco_train]
    cams = sorted(set(train_cams))

    # intra-camera pseudo-labels (deterministic from frozen feats) -> classifier sizes
    cl_same = {p for p in cl if len({index.camera_of(t) for t in p}) == 1}
    intra, n_cls = {}, {}
    by_cam = defaultdict(list)
    for t in loco_train:
        by_cam[index.camera_of(t)].append(t)
    for c, ts in by_cam.items():
        lab = ClusterAssigner(0.7, 10).assign(ts, np.stack([frozen_mean[t] for t in ts]), cl_same)
        intra.update(lab); n_cls[c] = ClusterAssigner.num_clusters(lab) or 1
    intra_pool = defaultdict(lambda: defaultdict(list))
    for t, l in intra.items():
        intra_pool[index.camera_of(t)][l].append(t)

    backbone = __import__("cowreid.encoder", fromlist=["DinoV2Backbone"]).DinoV2Backbone(
        model_name=VITB, pretrained=True).requires_grad_(False)
    backbone.unfreeze_last(args.n_blocks)
    model = FineTuneIICS(backbone, n_cls, args.proj_dim).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    start_step = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[resume] from step {start_step}", flush=True)

    inter = dbscan_cluster(loco_train, np.stack([frozen_mean[x] for x in loco_train]),
                           train_cams, eps=args.eps, cannot_link=cl)
    obj, mem = build_objective(args.proj_dim, max(1, num_clusters(inter)))
    obj.to(device)

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        if step % args.refresh_every == 0:
            E = embed_tids(model, cload, loco_train, args.T, device)
            inter = dbscan_cluster(loco_train, np.stack([E[t] for t in loco_train]),
                                   train_cams, eps=args.eps, cannot_link=cl)
            ce_emb = embed_crops_cached(model, cload, crop_paths, device)
            links, prec, ncand = mine_crop_ot_links(bags, ce_emb, index.tracklet_of,
                                                    min_conf=0.5, min_votes=3, gt=gt)
            inter = merge_labels(inter, links)
            mem.reset(max(1, num_clusters(inter)))
            print(f"  step {step}: #inter={num_clusters(inter)} crop-links={len(links)} "
                  f"prec={prec} ({time.time()-t0:.0f}s)", flush=True)

        if step % 2 == 0:                               # intra multi-branch CE
            cam = cams[int(rng.integers(len(cams)))]
            pool = [l for l, ts in intra_pool[cam].items() if ts]
            chosen = rng.choice(pool, size=min(args.P, len(pool)), replace=False)
            tids, labs = [], []
            for l in chosen:
                cand = intra_pool[cam][int(l)]
                pick = rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K)
                tids += pick.tolist(); labs += [int(l)] * len(pick)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = ce(model.logits(model.embed(cload.batch(tids, args.T)), cam),
                          torch.tensor(labs, device=device))
        else:                                           # inter cluster + topology
            pool = defaultdict(list)
            for t, l in inter.items():
                pool[l].append(t)
            chosen = rng.choice(list(pool), size=min(args.P, len(pool)), replace=False)
            tids = []
            for l in chosen:
                cand = pool[int(l)]
                tids += rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K).tolist()
            labs, pos, hard, clp = make_masks(tids, inter, cl)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model.embed(cload.batch(tids, args.T))
                loss, _ = obj(emb, positive_mask=pos.to(device), hard_negative_mask=hard.to(device),
                              cluster_labels=labs.to(device), cannot_link_pairs=clp.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT)
    print(f"[chunk] trained to step {step}/{args.target} in {time.time()-t0:.0f}s "
          f"(saved {CKPT})", flush=True)

    if args.eval and step >= args.target:
        gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
        gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
        query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
                 for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
        eval_tids = sorted({it.tracklet_id for it in query + gallery})
        emb = embed_tids(model, cload, eval_tids, args.frames, device)
        ev = ReIDEvaluator(ranks=(1, 5, 10))
        raw = ev.evaluate(query, gallery, emb)
        rr = evaluate_rerank(query, gallery, emb)
        print(f"\n>>> UNSUPERVISED ViT-B (step {step})", flush=True)
        print(f"    cosine : mAP={raw['mAP']:.3f} rank-1={raw['rank-1']:.3f} rank-5={raw['rank-5']:.3f}", flush=True)
        print(f"    rerank : mAP={rr['mAP']:.3f} rank-1={rr['rank-1']:.3f} rank-5={rr['rank-5']:.3f}", flush=True)


if __name__ == "__main__":
    main()
