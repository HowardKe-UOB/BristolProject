"""UNSUPERVISED ViT-B with CAMERA-AWARE PROXIES (CAP / O2CAP style), from scratch.

The champion uses Cluster-Contrast: ONE momentum centroid per cluster -> every
cross-camera instance of a cluster is pulled to a single point. A wrong crop-OT
merge then collapses two different cows into one centroid (this is what degraded
the two bootstrap / over-training runs). CAP instead splits each cluster into
per-(cluster,camera) PROXIES and uses two proxy-level losses:
  * intra-camera: softmax-CE over SAME-camera proxies, positive = own proxy
    (learns within-camera identity discrimination);
  * inter-camera: pull the instance toward the DIFFERENT-camera proxies that share
    its cluster (offline association) -- a SOFT cross-camera link that does not
    force a single centroid, so a wrong merge corrupts far less.
Everything else matches the champion (raw-space CA-Jaccard DBSCAN + crop-OT merge +
cannot-link topology to gate clustering; intra multi-branch CE branch kept).

Train a chunk:  python vitb_unsup_cap.py --wall 240 --target 1000
Then evaluate:  python eval_ckpt.py --ckpt _vitb_cap_ckpt.pt --tag cap
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags, mine_crop_ot_links
from cowreid.encoder import DinoV2Backbone
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import FineTuneIICS, merge_labels
from vitb_unsup import (CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader,
                        embed_crops_cached, embed_tids)

CKPT = "_vitb_cap_ckpt.pt"


class ProxyMemory:
    """Per-(cluster,camera) proxy bank with momentum update (CAP)."""

    def __init__(self, temp=0.07, momentum=0.2):
        self.temp = temp; self.m = momentum

    def build(self, tids, emb, inter, cam_of, device):
        groups = defaultdict(list)
        for t in tids:
            groups[(inter[t], cam_of(t))].append(t)
        self.keys = sorted(groups)
        self.key2idx = {k: i for i, k in enumerate(self.keys)}
        P, pc, pcam = [], [], []
        for k in self.keys:
            v = np.mean([emb[t] for t in groups[k]], axis=0)
            P.append(v / (np.linalg.norm(v) + 1e-12)); pc.append(k[0]); pcam.append(k[1])
        self.proxies = F.normalize(torch.tensor(np.array(P), dtype=torch.float32,
                                                device=device), dim=1)
        self.pcluster = np.array(pc); self.pcamera = np.array(pcam)
        self.tid2proxy = {t: self.key2idx[(inter[t], cam_of(t))] for t in tids}
        self.members = {i: groups[k] for i, k in enumerate(self.keys)}
        # precompute, per proxy, the same-camera proxy set and same-cluster cross-cam set
        self.same_cam = [np.where(self.pcamera == self.pcamera[i])[0] for i in range(len(self.keys))]
        self.pos_inter = []
        for i in range(len(self.keys)):
            diff = np.where(self.pcamera != self.pcamera[i])[0]
            self.pos_inter.append((diff, diff[self.pcluster[diff] == self.pcluster[i]]))

    @torch.no_grad()
    def update(self, feats, pidx):
        for f, p in zip(feats, pidx):
            self.proxies[p] = F.normalize(self.m * self.proxies[p] + (1 - self.m) * f, dim=0)


def cap_loss(feats, pidx, mem):
    """CAP intra + inter camera proxy loss for a batch."""
    sim = feats @ mem.proxies.t() / mem.temp                # (B, nP)
    losses = []
    for i in range(feats.shape[0]):
        p = pidx[i]
        sc = mem.same_cam[p]
        tgt = int(np.where(sc == p)[0][0])
        l_intra = F.cross_entropy(sim[i, sc].unsqueeze(0),
                                  torch.tensor([tgt], device=feats.device))
        diff, pos = mem.pos_inter[p]
        if len(pos) > 0:
            num = torch.logsumexp(sim[i, pos], dim=0)
            den = torch.logsumexp(sim[i, diff], dim=0)
            l_inter = -(num - den)
        else:
            l_inter = torch.zeros((), device=feats.device)
        losses.append(l_intra + l_inter)
    return torch.stack(losses).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    device = "cuda"; t0 = time.time()
    ckpt_path = args.ckpt

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(args.seed))

    train_cams_all = [c for c in {t.camera for t in tracklets} if c != HOLD]
    bags, _ = crossview_crop_bags(Manifest([s for t in tracklets for s in t.samples]),
                                  topo, train_cams_all, index, max_bags=args.max_bags)
    crop_paths = sorted({p for _a, _b, A, B in bags for p in (A + B)})

    d = np.load(args.vits_cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    loco_train = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    train_cams = [cam_of(t) for t in loco_train]
    cams = sorted(set(train_cams))

    cl_same = {p for p in cl if len({cam_of(t) for t in p}) == 1}
    intra, n_cls = {}, {}
    by_cam = defaultdict(list)
    for t in loco_train:
        by_cam[cam_of(t)].append(t)
    for c, ts in by_cam.items():
        lab = ClusterAssigner(0.7, 10).assign(ts, np.stack([frozen_mean[t] for t in ts]), cl_same)
        intra.update(lab); n_cls[c] = ClusterAssigner.num_clusters(lab) or 1
    intra_pool = defaultdict(lambda: defaultdict(list))
    for t, l in intra.items():
        intra_pool[cam_of(t)][l].append(t)

    backbone = DinoV2Backbone(model_name=VITB, pretrained=True).requires_grad_(False)
    backbone.unfreeze_last(args.n_blocks)
    model = FineTuneIICS(backbone, n_cls, args.proj_dim).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError):
            pass
        start_step = ck["step"]
        print(f"[resume] step {start_step}", flush=True)

    mem = ProxyMemory(temp=0.07)
    # proxies must live in the MODEL embedding space (256-d), not the 384-d ViT-S
    # frozen features; embed once at init so resume between refreshes is consistent.
    E0 = embed_tids(model, cload, loco_train, args.T, device)
    cluster_feat = frozen_mean if start_step == 0 else E0
    inter = dbscan_cluster(loco_train, np.stack([cluster_feat[t] for t in loco_train]),
                           train_cams, eps=args.eps, cannot_link=cl)
    mem.build(loco_train, E0, inter, cam_of, device)
    print(f"[init] clusters={num_clusters(inter)} proxies={len(mem.keys)}", flush=True)

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        if step % args.refresh_every == 0:
            E = embed_tids(model, cload, loco_train, args.T, device)
            inter = dbscan_cluster(loco_train, np.stack([E[t] for t in loco_train]),
                                   train_cams, eps=args.eps, cannot_link=cl)
            ce_emb = embed_crops_cached(model, cload, crop_paths, device)
            links, prec, _ = mine_crop_ot_links(bags, ce_emb, index.tracklet_of,
                                                min_conf=0.5, min_votes=3, gt=gt)
            inter = merge_labels(inter, links)
            mem.build(loco_train, E, inter, cam_of, device)
            print(f"  step {step}: clusters={num_clusters(inter)} proxies={len(mem.keys)} "
                  f"links={len(links)} prec={prec} ({time.time()-t0:.0f}s)", flush=True)

        if step % 2 == 0:                              # intra multi-branch CE (kept)
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
        else:                                          # CAP proxy loss (inter+intra camera)
            pidx_pool = list(mem.members)
            chosen = rng.choice(pidx_pool, size=min(args.P, len(pidx_pool)), replace=False)
            tids, pidx = [], []
            for pi in chosen:
                cand = mem.members[int(pi)]
                pick = rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K)
                tids += pick.tolist(); pidx += [int(pi)] * len(pick)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model.embed(cload.batch(tids, args.T))
                loss = cap_loss(emb.float(), pidx, mem)
            mem.update(emb.detach().float(), pidx)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt_path)
    print(f"[chunk] -> step {step}/{args.target} in {time.time()-t0:.0f}s (saved {ckpt_path})", flush=True)


if __name__ == "__main__":
    main()
