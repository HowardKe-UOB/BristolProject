"""STRICT high-precision cross-camera mining, warm-started from the champion ckpt.

The two earlier bottleneck attacks RAISED recall at fixed ~15% link precision and
regressed. This does the OPPOSITE: adds a cosine gate to the crop-OT must-link
mining so only high-similarity (mostly correct) links survive (diagnostic:
cos>=0.5 cuts false links 108->52 while keeping 19/22 true links). Everything else
is identical to the champion run (raw-space CA-Jaccard DBSCAN, emb256 crop mining).

Fair test = two continued-training runs from a COPY of the champion checkpoint:
  * CONTROL (--cos-thr -1, --ckpt _vitb_ctrl_ckpt.pt): loose mining -> isolates
    warm-start drift.
  * STRICT  (--cos-thr 0.5, --ckpt _vitb_strict_ckpt.pt): cosine-gated mining.
If STRICT > CONTROL after the same #steps, cleaner links help.

    python vitb_unsup_strict.py --ckpt _vitb_strict_ckpt.pt --cos-thr 0.5 --wall 240 --target 1300
    python vitb_unsup_strict.py --ckpt _vitb_strict_ckpt.pt --cos-thr 0.5 --wall 240 --target 1300 --eval
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import json
import os
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.sinkhorn import match_with_dustbin
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import FineTuneIICS, make_masks, merge_labels
from train_phase2 import build_objective
from vitb_unsup import (CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader,
                        embed_crops_cached, embed_tids)

CHAMP = "_vitb_unsup_ckpt.pt"


def mine_gated(bags, crop_emb, path_to_tracklet, min_conf, min_votes, cos_thr, gt):
    votes = defaultdict(float); counts = defaultdict(int)
    for _cA, _cB, A, B in bags:
        EA = np.stack([crop_emb[p] for p in A]); EB = np.stack([crop_emb[p] for p in B])
        EA = EA / (np.linalg.norm(EA, axis=1, keepdims=True) + 1e-12)
        EB = EB / (np.linalg.norm(EB, axis=1, keepdims=True) + 1e-12)
        sim = EA @ EB.T
        for i, j, conf in match_with_dustbin(1.0 - sim, eps=0.1):
            if conf < min_conf or sim[i, j] < cos_thr:
                continue
            ta, tb = path_to_tracklet(A[i]), path_to_tracklet(B[j])
            if ta and tb and ta != tb:
                k = frozenset((ta, tb)); votes[k] += conf; counts[k] += 1
    links = [k for k in votes if counts[k] >= min_votes]
    prec = float(np.mean([gt[tuple(k)[0]] == gt[tuple(k)[1]] for k in links])) if links else None
    return links, prec, len(votes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cos-thr", type=float, default=0.5)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--target", type=int, default=1300)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--refresh-every", type=int, default=150)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    args = ap.parse_args()
    device = "cuda"; t0 = time.time()

    if not os.path.exists(args.ckpt):
        shutil.copy(CHAMP, args.ckpt)
        print(f"[init] copied champion -> {args.ckpt}", flush=True)

    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

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
    train_cams = [index.camera_of(t) for t in loco_train]
    cams = sorted(set(train_cams))

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

    backbone = DinoV2Backbone(model_name=VITB, pretrained=True).requires_grad_(False)
    backbone.unfreeze_last(args.n_blocks)
    model = FineTuneIICS(backbone, n_cls, args.proj_dim).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    try:
        opt.load_state_dict(ck["opt"])
    except (ValueError, KeyError):
        pass
    step = ck["step"]
    print(f"[resume] {args.ckpt} step {step} cos_thr={args.cos_thr}", flush=True)

    inter = dbscan_cluster(loco_train, np.stack([frozen_mean[x] for x in loco_train]),
                           train_cams, eps=args.eps, cannot_link=cl)
    obj, mem = build_objective(args.proj_dim, max(1, num_clusters(inter)))
    obj.to(device)

    while step < args.target and (time.time() - t0) < args.wall:
        if step % args.refresh_every == 0:
            E = embed_tids(model, cload, loco_train, args.T, device)
            inter = dbscan_cluster(loco_train, np.stack([E[t] for t in loco_train]),
                                   train_cams, eps=args.eps, cannot_link=cl)
            ce_emb = embed_crops_cached(model, cload, crop_paths, device)   # emb256 (champion)
            links, prec, ncand = mine_gated(bags, ce_emb, index.tracklet_of,
                                            args.min_conf, args.min_votes, args.cos_thr, gt)
            inter = merge_labels(inter, links)
            mem.reset(max(1, num_clusters(inter)))
            print(f"  step {step}: #inter={num_clusters(inter)} links={len(links)} "
                  f"prec={prec} ({time.time()-t0:.0f}s)", flush=True)

        if step % 2 == 0:
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
        else:
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

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, args.ckpt)
    print(f"[chunk] -> step {step}/{args.target} in {time.time()-t0:.0f}s", flush=True)

    if args.eval and step >= args.target:
        import torch.nn.functional as F
        from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
        from cowreid.eval import _score
        from cowreid.st_inference import INF, build_st_mask

        gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
        gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
        query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
                 for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
        q, g = list(query), list(gallery)
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        eval_tids = sorted({it.tracklet_id for it in q + g})
        mask = build_st_mask(q, g, index, topo, margin=0)

        @torch.no_grad()
        def feat768(tids, flip=False):
            model.eval(); out = {}
            for i in range(0, len(tids), 12):
                ch = tids[i:i + 12]; x = cload.batch(ch, args.frames, train=False)
                if flip:
                    x = torch.flip(x, dims=[-1])
                with torch.autocast("cuda", dtype=torch.float16):
                    e = F.normalize(model.head.pool(model._frames(x)), dim=1)
                for t, v in zip(ch, e.float().cpu().numpy()):
                    out[t] = v
            return out
        E0 = feat768(eval_tids); E1 = feat768(eval_tids, flip=True)
        n = lambda v: v / (np.linalg.norm(v) + 1e-12)
        Etta = {t: n(n(E0[t]) + n(E1[t])) for t in eval_tids}

        def champ(E):
            Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
            X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
            Qcc, Gcc = cc[:len(q)], cc[len(q):]
            Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
            return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                        dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)
        report = {"ckpt": args.ckpt, "cos_thr": args.cos_thr, "step": int(step)}
        print(f"\n>>> {args.ckpt} (cos_thr={args.cos_thr}, step {step}), leave-out {HOLD}", flush=True)
        for tag, E in [("champion recipe", E0), ("+ flip TTA", Etta)]:
            dist = champ(E)
            r = _score(q, g, dist, (1, 5, 10)); dm = dist.copy(); dm[mask] = INF
            rs = _score(q, g, dm, (1, 5, 10))
            print(f"    {tag:18s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
                  f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
            report[tag] = {"plain": r, "st": rs}
        outp = f"artifacts2/strict_eval_{os.path.basename(args.ckpt).replace('.pt','')}_v1.json"
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"saved {outp}   (champion ref: no-TTA 0.706, +TTA 0.718)", flush=True)


if __name__ == "__main__":
    main()
