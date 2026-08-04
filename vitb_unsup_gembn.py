"""UNSUPERVISED ViT-B with a retrieval-oriented head: spatial GeM pooling over
DINOv2 patch tokens + temporal GeM over frames + BNNeck, replacing the CLS-token /
attention-pool / 256-d projection stack. FULLY LABEL-FREE -- identical unsupervised
pipeline as vitb_unsup.py (per-camera CE on intra-camera pseudo-labels, CA-Jaccard
DBSCAN inter-camera clusters, crop-OT cross-view mining, topology cannot-link,
Cluster-Contrast), only the architecture changes.

Why this should help retrieval:
  * patch-token GeM > CLS token: aggregates the whole coat pattern, not one global
    token (standard strong ViT retrieval descriptor).
  * BNNeck (Luo 2019): a bias-free BatchNorm makes the 768-d feature directly
    metric/cosine-friendly and balances the per-camera classifier -- this is the
    principled version of the "retrieve on the backbone feature" trick we found by
    hand.
  * GeM temporal pooling: emphasises the most confident frames over a flat mean.

Reuses the uint8 image cache + checkpoint-resume machinery of vitb_unsup.py.

Train a chunk:  python vitb_unsup_gembn.py --wall 240 --target 1000
Final eval:     python vitb_unsup_gembn.py --wall 240 --target 1000 --eval
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
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags, mine_crop_ot_links
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import make_masks, merge_labels
from train_phase2 import build_objective
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, IMG, VITB, CacheLoader

CKPT = "_vitb_gembn_ckpt.pt"
EMB_NPZ = "_vitb_gembn_emb_v1.npz"


class GeMBackbone(nn.Module):
    """DINOv2 ViT with spatial GeM pooling over patch tokens (learnable p)."""

    def __init__(self, model_name=VITB, n_blocks=4, p_init=3.0):
        super().__init__()
        base = DinoV2Backbone(model_name=model_name, pretrained=True).requires_grad_(False)
        base.unfreeze_last(n_blocks)
        self.model = base.model
        self.embed_dim = base.embed_dim
        self.num_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        self.p = nn.Parameter(torch.ones(1) * p_init)

    def forward(self, x):                          # (N, 3, H, W) -> (N, D)
        tok = self.model.forward_features(x)       # (N, 1+P, D), token 0 = CLS
        patch = tok[:, self.num_prefix:, :].clamp(min=1e-6)
        p = self.p.clamp(min=1.0, max=6.0)
        return patch.pow(p).mean(dim=1).pow(1.0 / p)

    def trainable_parameters(self):
        return [q for q in self.model.parameters() if q.requires_grad] + [self.p]


class TemporalGeM(nn.Module):
    def __init__(self, p_init=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)

    def forward(self, feats):                      # (B, T, D) -> (B, D)
        p = self.p.clamp(min=1.0, max=6.0)
        return feats.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)


class GeMBNReID(nn.Module):
    """GeM backbone + temporal GeM + BNNeck + per-camera cosine classifiers.
    ``embed`` returns the L2-normalised BNNeck feature -- used for retrieval,
    cluster memory, contrastive and cannot-link losses alike."""

    def __init__(self, backbone: GeMBackbone, n_classes_per_cam, temporal_p=3.0):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim
        self.tpool = TemporalGeM(temporal_p)
        self.bnneck = nn.BatchNorm1d(self.embed_dim)
        self.bnneck.bias.requires_grad_(False)     # BNNeck: no bias
        self._key = {c: c.replace(".", "_") for c in n_classes_per_cam}
        self.classifiers = nn.ModuleDict(
            {self._key[c]: nn.Linear(self.embed_dim, max(1, k), bias=False)
             for c, k in n_classes_per_cam.items()})

    def _frames(self, clips):                      # (B, T, 3, H, W) -> (B, T, D)
        B, T = clips.shape[:2]
        return self.backbone(clips.flatten(0, 1)).view(B, T, -1)

    def embed(self, clips):
        pooled = self.tpool(self._frames(clips))
        return F.normalize(self.bnneck(pooled), dim=1)

    def logits(self, emb, camera, scale: float = 16.0):
        w = F.normalize(self.classifiers[self._key[camera]].weight, dim=1)
        return scale * F.linear(emb, w)

    def all_scores(self, emb, scale: float = 16.0):
        return torch.cat([F.softmax(scale * F.linear(emb, F.normalize(c.weight, dim=1)), dim=1)
                          for c in self.classifiers.values()], dim=1)


@torch.no_grad()
def embed_tids(model, cload, tids, T, device, bs=12):
    model.eval(); out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.embed(cload.batch(chunk, T, train=False))
        for t, v in zip(chunk, e.float().cpu().numpy()):
            out[t] = v
    return out


@torch.no_grad()
def embed_crops_cached(model, cload, paths, device, bs=48):
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
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts2/gembn_eval_v1.json")
    args = ap.parse_args()
    device = "cuda"
    t0 = time.time()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    cache = np.load(CACHE_NPY, mmap_mode="r")
    p2r = json.load(open(CACHE_JSON))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(args.seed))

    # crop-OT bags (cross-view mining) on train cameras
    train_manifest = Manifest([s for t in tracklets for s in t.samples])
    train_cams_all = [c for c in {t.camera for t in tracklets} if c != HOLD]
    bags, _crop_paths = crossview_crop_bags(train_manifest, topo, train_cams_all, index,
                                            max_bags=args.max_bags)
    crop_paths = sorted({p for _a, _b, A, B in bags for p in (A + B)})

    d = np.load(args.vits_cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    loco_train = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    train_cams = [index.camera_of(t) for t in loco_train]
    cams = sorted(set(train_cams))

    # intra-camera pseudo-labels from frozen feats -> per-camera classifier sizes
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

    backbone = GeMBackbone(model_name=VITB, n_blocks=args.n_blocks)
    model = GeMBNReID(backbone, n_cls).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": [p for n, p in model.named_parameters()
                     if not n.startswith("backbone.")], "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    start_step = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[resume] from step {start_step}", flush=True)

    D = model.embed_dim
    inter = dbscan_cluster(loco_train, np.stack([frozen_mean[x] for x in loco_train]),
                           train_cams, eps=args.eps, cannot_link=cl)
    obj, mem = build_objective(D, max(1, num_clusters(inter)))
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
                  f"prec={prec} p_sp={float(backbone.p):.2f} p_t={float(model.tpool.p):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % 2 == 0:                          # intra multi-branch CE
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
        else:                                      # inter cluster + topology
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
        from new_levers import (camera_center, dist_cosine, dist_rerank, pca_whiten, rrf)
        from cowreid.eval import _score
        from cowreid.st_inference import INF, build_st_mask

        gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
        gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
        query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
                 for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
        q, g = list(query), list(gallery)
        eval_tids = sorted({it.tracklet_id for it in q + g})
        Emb = embed_tids(model, cload, eval_tids, args.frames, device)
        np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                            feat768=np.stack([Emb[t] for t in eval_tids]))
        print(f"saved {EMB_NPZ}", flush=True)

        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        mask = build_st_mask(q, g, index, topo, margin=0)
        Qf = np.stack([Emb[it.tracklet_id] for it in q])
        Gf = np.stack([Emb[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf])
        cc = camera_center(q + g, X); Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        variants = {
            "cosine": dist_cosine(Qf, Gf),
            "CC": dist_cosine(Qcc, Gcc),
            "PCAW": dist_cosine(Qw, Gw),
            "CC-RR": dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6),
        }
        variants["RRF(CC,PCAW,CC-RR)"] = rrf([variants["CC"], variants["PCAW"], variants["CC-RR"]], k=20)
        report = {"checkpoint_step": int(step), "arch": "GeM+BNNeck"}
        print(f"\n>>> UNSUPERVISED GeM+BNNeck (step {step}), leave-out {HOLD}", flush=True)
        for name, dist in variants.items():
            r = _score(q, g, dist, (1, 5, 10))
            dm = dist.copy(); dm[mask] = INF
            rs = _score(q, g, dm, (1, 5, 10))
            print(f"    {name:20s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
                  f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
            report[name] = {"plain": r, "st": rs}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
