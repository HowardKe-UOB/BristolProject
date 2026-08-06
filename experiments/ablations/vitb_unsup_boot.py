"""Bootstrap the UNSUPERVISED cross-camera association (the true bottleneck) by
generating pseudo-labels in the CAMERA-INVARIANT feature space.

Key idea: we found that per-camera centering (CC) makes the 768-d feature far more
cross-camera discriminative at retrieval (rank-1 0.62->0.66, and 0.71 fused). The
training pipeline, however, still CLUSTERS and MINES cross-view links in the RAW
(camera-biased) space, capping pseudo-label precision (~0.13-0.16) -> the same cow
stays split across cameras. Here we camera-center the embeddings *before* both the
CA-Jaccard DBSCAN clustering and the crop-OT must-link mining, so the pseudo-labels
merge cross-camera positives that the raw space kept apart. Better labels -> the raw
features become more camera-invariant -> re-mine -> iterate. FULLY LABEL-FREE (CC is
a label-free transform); the training loss still runs on the raw model embeddings.

Warm-starts from the best CLS-token checkpoint (`_vitb_boot_ckpt.pt`, a copy of
`_vitb_unsup_ckpt.pt` at step 1000) and continues.

Train a chunk:  python vitb_unsup_boot.py --wall 240 --target 1600
Final eval:     python vitb_unsup_boot.py --wall 240 --target 1600 --eval
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

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags, mine_crop_ot_links
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import FineTuneIICS, make_masks, merge_labels
from train_phase2 import build_objective
from vitb_unsup import (CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader,
                        embed_crops_cached, embed_tids)

CKPT = "_vitb_boot_ckpt.pt"
EMB_NPZ = "_vitb_boot_emb_v1.npz"


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cc_tracklets(tids, E, cam_of, strength=1.0):
    """Per-camera-centered copy of the tracklet embeddings (label-free)."""
    X = _l2(np.stack([E[t] for t in tids]))
    out = X.copy()
    by_cam = defaultdict(list)
    for i, t in enumerate(tids):
        by_cam[cam_of(t)].append(i)
    for _c, idx in by_cam.items():
        out[idx] = X[idx] - strength * X[idx].mean(0, keepdims=True)
    out = _l2(out)
    return {t: out[i] for i, t in enumerate(tids)}


def cc_crops(crop_emb, strength=1.0):
    """Per-camera-centered copy of crop embeddings; camera parsed from the path
    (2025Sep18/<camera>/<gt>/<ts>.jpeg)."""
    paths = list(crop_emb)
    X = _l2(np.stack([crop_emb[p] for p in paths]))
    out = X.copy()
    by_cam = defaultdict(list)
    for i, p in enumerate(paths):
        by_cam[p.replace("\\", "/").split("/")[-3]].append(i)
    for _c, idx in by_cam.items():
        out[idx] = X[idx] - strength * X[idx].mean(0, keepdims=True)
    out = _l2(out)
    return {p: out[i] for i, p in enumerate(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--refresh-every", type=int, default=200)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--cc-strength", type=float, default=1.0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts2/boot_eval_v1.json")
    args = ap.parse_args()
    device = "cuda"
    t0 = time.time()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of

    cache = np.load(CACHE_NPY, mmap_mode="r")
    p2r = json.load(open(CACHE_JSON))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(args.seed))

    train_manifest = Manifest([s for t in tracklets for s in t.samples])
    train_cams_all = [c for c in {t.camera for t in tracklets} if c != HOLD]
    bags, _cp = crossview_crop_bags(train_manifest, topo, train_cams_all, index,
                                    max_bags=args.max_bags)
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
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device)
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError):
            print("[warn] opt state not loaded (fresh optimizer)", flush=True)
        start_step = ck["step"]
        print(f"[resume] from step {start_step}", flush=True)

    # ---- initial pseudo-labels in the CAMERA-CENTERED space ----------------- #
    fm_cc = cc_tracklets(loco_train, frozen_mean, cam_of, args.cc_strength)
    inter = dbscan_cluster(loco_train, np.stack([fm_cc[x] for x in loco_train]),
                           train_cams, eps=args.eps, cannot_link=cl)
    print(f"[init] CC-space #inter={num_clusters(inter)} (raw-space ref ~90)", flush=True)
    obj, mem = build_objective(args.proj_dim, max(1, num_clusters(inter)))
    obj.to(device)

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        if step % args.refresh_every == 0:
            E = embed_tids(model, cload, loco_train, args.T, device)
            E_cc = cc_tracklets(loco_train, E, cam_of, args.cc_strength)          # <-- CC space
            inter = dbscan_cluster(loco_train, np.stack([E_cc[t] for t in loco_train]),
                                   train_cams, eps=args.eps, cannot_link=cl)
            ce_emb = embed_crops_cached(model, cload, crop_paths, device)
            ce_cc = cc_crops(ce_emb, args.cc_strength)                            # <-- CC space
            links, prec, ncand = mine_crop_ot_links(bags, ce_cc, index.tracklet_of,
                                                    min_conf=0.5, min_votes=3, gt=gt)
            inter = merge_labels(inter, links)
            mem.reset(max(1, num_clusters(inter)))
            print(f"  step {step}: #inter={num_clusters(inter)} crop-links={len(links)} "
                  f"prec={prec} ({time.time()-t0:.0f}s)", flush=True)

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
        from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
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
        report = {"checkpoint_step": int(step), "method": "bootstrap CC-space pseudo-labels"}
        print(f"\n>>> UNSUPERVISED BOOTSTRAP (step {step}), leave-out {HOLD}", flush=True)
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
