"""SUPERVISED ViT-B twin of vitb_unsup.py -- same cache/chunk/resume machinery,
same hyper-parameters and step budget, but inter-camera labels are GROUND TRUTH
(fixed; no clustering refresh, no mining). Purpose: a *paired* supervised
reference for the inference-lever study (emb256 vs feat768 x CC/ST/RR), so the
unsupervised gains from evaluating the backbone feature are compared against a
supervised model given the identical treatment.

Train a chunk:      python vitb_sup.py --wall 480 --target 1000
Final eval:         python vitb_sup.py --wall 480 --target 1000 --eval
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
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from train_finetune_iics import FineTuneIICS, make_masks
from train_phase2 import build_objective
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader

CKPT = "_vitb_sup_ckpt.pt"
EMB_NPZ = "_vitb_sup_emb_v1.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--vits-cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=480)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts2/st_inference_vitb_sup_v1.json")
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

    d = np.load(args.vits_cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    loco_train = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    cams = sorted({index.camera_of(t) for t in loco_train})

    # intra-camera pseudo-labels (same procedure as the unsupervised twin / push_sota)
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

    # inter labels = GROUND TRUTH identities (fixed for the whole run)
    ids = sorted({gt[t] for t in loco_train})
    idmap = {g: i for i, g in enumerate(ids)}
    inter = {t: idmap[gt[t]] for t in loco_train}
    print(f"[supervised] {len(loco_train)} train tracklets, {len(ids)} identities", flush=True)

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
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[resume] from step {start_step}", flush=True)

    obj, mem = build_objective(args.proj_dim, len(ids))
    obj.to(device)

    pool = defaultdict(list)
    for t, l in inter.items():
        pool[l].append(t)

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        if step % args.refresh_every == 0:
            mem.reset(len(ids))                      # same cadence as the unsup twin
            print(f"  step {step}: memory reset ({time.time()-t0:.0f}s)", flush=True)

        if step % 2 == 0:                            # intra multi-branch CE
            cam = cams[int(rng.integers(len(cams)))]
            pl = [l for l, ts in intra_pool[cam].items() if ts]
            chosen = rng.choice(pl, size=min(args.P, len(pl)), replace=False)
            tids, labs = [], []
            for l in chosen:
                cand = intra_pool[cam][int(l)]
                pick = rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K)
                tids += pick.tolist(); labs += [int(l)] * len(pick)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = ce(model.logits(model.embed(cload.batch(tids, args.T)), cam),
                          torch.tensor(labs, device=device))
        else:                                        # inter: GT-label cluster objective
            chosen = rng.choice(list(pool), size=min(args.P, len(pool)), replace=False)
            tids = []
            for l in chosen:
                cand = pool[int(l)]
                tids += rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K).tolist()
            labs, posm, hard, clp = make_masks(tids, inter, cl)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model.embed(cload.batch(tids, args.T))
                loss, _ = obj(emb, positive_mask=posm.to(device), hard_negative_mask=hard.to(device),
                              cluster_labels=labs.to(device), cannot_link_pairs=clp.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT)
    print(f"[chunk] trained to step {step}/{args.target} in {time.time()-t0:.0f}s "
          f"(saved {CKPT})", flush=True)

    if args.eval and step >= args.target:
        from st_eval_vitb import embed_both
        from st_validate2 import run_all

        gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
        gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
        query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
                 for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
        eval_tids = sorted({it.tracklet_id for it in query + gallery})
        E, Ft = embed_both(model, cload, eval_tids, args.frames, device)
        np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                            emb256=np.stack([E[t] for t in eval_tids]),
                            feat768=np.stack([Ft[t] for t in eval_tids]))
        print(f"saved {EMB_NPZ}", flush=True)
        report = {"checkpoint_step": step, "mode": "supervised"}
        print(f"\n=== ViT-B SUPERVISED, 256-d projected embedding (leave-out {HOLD}) ===", flush=True)
        run_all(query, gallery, E, index, topo, margin=0, tag="emb256/", report=report)
        print(f"\n=== ViT-B SUPERVISED, 768-d pooled backbone feature ===", flush=True)
        run_all(query, gallery, Ft, index, topo, margin=0, tag="feat768/", report=report)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
