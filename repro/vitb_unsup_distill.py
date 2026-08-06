"""ENSEMBLE-DISTILLED student (label-free): FIXED teacher pseudo-labels from the
5-seed ensemble space + NON-TRANSITIVE pairwise cross-camera links.

Design (motivated by the label-quality diagnostics in distill_diag.py /
make_distill_labels.py):
  * intra-camera clusters computed ONCE in the ensemble space (pairwise precision
    0.879 -- the cleanest labels the pipeline has ever had). One CAP proxy per
    cluster (clusters are single-camera). FIXED all run -- no refresh, no drift,
    no over-merge collapse.
  * cross-camera signal = the 103 ensemble mutual-NN (k=1) links at 60% precision,
    used as PAIRWISE positive pairs between proxies (pull instance toward the
    linked other-camera proxy, softmax over all other-camera proxies). NOT merged
    into clusters -- transitive merging collapses 60% link precision to ~17% pair
    precision, pairwise use keeps the full 60%.
  * intra multi-branch CE branch kept (per-camera classifiers from the same fixed
    clusters). Early-stopped at --target (over-training lesson).

Train:  python vitb_unsup_distill.py --seed 5 --ckpt _vitb_dst_s5_ckpt.pt --wall 240 --target 1000
Eval:   python eval_ckpt.py --ckpt _vitb_dst_s5_ckpt.pt --tag dst_s5
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
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.encoder import DinoV2Backbone
from cowreid.tracklets import TrackletIndex
from consensus_ens import mutual_knn_links
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader, embed_tids

ENS_NPZ = "_vitb_cap_ens5_emb_v1.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--link-k", type=int, default=1)
    ap.add_argument("--w-link", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--momentum", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ckpt", default="_vitb_dst_s5_ckpt.pt")
    args = ap.parse_args()
    device = "cuda"; t0 = time.time()

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    cam_of = index.camera_of

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(args.seed))

    # ---------- FIXED teacher structures from the ensemble space ---------- #
    d = np.load(ENS_NPZ, allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    seeds = [k for k in d.files if k != "ids"]
    Xens = np.mean([d[s] for s in seeds], axis=0)
    Xens = Xens / (np.linalg.norm(Xens, axis=1, keepdims=True) + 1e-12)

    g_tids = [t.tracklet_id for t in tracklets if t.camera != HOLD]
    cams_list = [cam_of(t) for t in g_tids]
    cams = sorted(set(cams_list))

    cl_same = {p for p in cl if len({cam_of(t) for t in p}) == 1}
    by_cam = defaultdict(list)
    for t in g_tids:
        by_cam[cam_of(t)].append(t)
    intra_global, n_cls, intra_local = {}, {}, {}
    off = 0
    for c, ts in sorted(by_cam.items()):
        E = np.stack([Xens[pos[t]] for t in ts])
        lab = ClusterAssigner(0.7, 10).assign(ts, E, cl_same)
        for t in ts:
            intra_local[t] = lab[t]
            intra_global[t] = off + lab[t]
        k = ClusterAssigner.num_clusters(lab) or 1
        n_cls[c] = k; off += k
    n_proxy = off
    print(f"[teacher] {n_proxy} fixed intra-camera proxies", flush=True)

    X = np.stack([Xens[pos[t]] for t in g_tids]).astype(np.float32)
    links = mutual_knn_links(X.copy(), cams_list, k=args.link_k)
    # proxy-level links with confidence = cosine of the tracklet pair
    plinks = defaultdict(list)                        # proxy -> [(other_proxy, conf)]
    linked_tids = set()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    for a, b in (tuple(l) for l in links):
        ca, cb = intra_global[g_tids[a]], intra_global[g_tids[b]]
        conf = float(Xn[a] @ Xn[b])
        plinks[ca].append((cb, conf)); plinks[cb].append((ca, conf))
        linked_tids.add(g_tids[a]); linked_tids.add(g_tids[b])
    print(f"[teacher] {len(links)} mutual-NN k={args.link_k} links "
          f"({len(linked_tids)} tracklets touched)", flush=True)

    proxy_members = defaultdict(list)
    for t in g_tids:
        proxy_members[intra_global[t]].append(t)
    proxy_cam = {}
    for t in g_tids:
        proxy_cam[intra_global[t]] = cam_of(t)
    pcam_arr = np.array([proxy_cam[i] for i in range(n_proxy)])
    linked_pool = sorted({p for p in plinks})

    intra_pool = defaultdict(lambda: defaultdict(list))
    for t in g_tids:
        intra_pool[cam_of(t)][intra_local[t]].append(t)

    # ---------- model ---------- #
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
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except (ValueError, KeyError):
            pass
        start_step = ck["step"]
        print(f"[resume] step {start_step}", flush=True)

    # proxy bank init from student embeddings of the fixed clusters
    E0 = embed_tids(model, cload, g_tids, args.T, device)
    P0 = np.stack([np.mean([E0[t] for t in proxy_members[i]], axis=0) for i in range(n_proxy)])
    P0 = P0 / (np.linalg.norm(P0, axis=1, keepdims=True) + 1e-12)
    proxies = F.normalize(torch.tensor(P0, dtype=torch.float32, device=device), dim=1)
    same_cam_idx = {c: np.where(pcam_arr == c)[0] for c in cams}
    diff_cam_idx = {c: np.where(pcam_arr != c)[0] for c in cams}

    def proxy_update(feats, pidx):
        with torch.no_grad():
            for f, p in zip(feats, pidx):
                proxies[p] = F.normalize(args.momentum * proxies[p]
                                         + (1 - args.momentum) * f, dim=0)

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        mode = step % 2
        if mode == 0:                                  # intra multi-branch CE
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
        else:                                          # proxy CE + pairwise link loss
            # half the sampled proxies come from the linked pool so links fire
            n_link = min(args.P // 2, len(linked_pool))
            chosen = list(rng.choice(linked_pool, size=n_link, replace=False))
            others = rng.choice(n_proxy, size=args.P - n_link, replace=False)
            chosen += [int(x) for x in others]
            tids, pidx = [], []
            for pi in chosen:
                cand = proxy_members[int(pi)]
                pick = rng.choice(cand, size=min(args.K, len(cand)), replace=len(cand) < args.K)
                tids += pick.tolist(); pidx += [int(pi)] * len(pick)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model.embed(cload.batch(tids, args.T))
            embf = emb.float()
            # clone: the in-place momentum update must not touch tensors saved
            # for backward
            sim = embf @ proxies.clone().t() / args.temp    # (B, nP)
            l_items = []
            for i in range(embf.shape[0]):
                p = pidx[i]; c = proxy_cam[p]
                sc = same_cam_idx[c]
                tgt = int(np.where(sc == p)[0][0])
                l_i = F.cross_entropy(sim[i, sc].unsqueeze(0),
                                      torch.tensor([tgt], device=device))
                if p in plinks:                        # pairwise cross-camera pull
                    dc = diff_cam_idx[c]
                    den = torch.logsumexp(sim[i, dc], dim=0)
                    l_link = 0.0; wsum = 0.0
                    for (op, conf) in plinks[p]:
                        l_link = l_link + conf * (den - sim[i, op])
                        wsum += conf
                    l_i = l_i + args.w_link * l_link / max(wsum, 1e-9)
                l_items.append(l_i)
            loss = torch.stack(l_items).mean()
            proxy_update(embf.detach(), pidx)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1
        if step % 200 == 0:
            print(f"  step {step} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, args.ckpt)
    print(f"[chunk] -> step {step}/{args.target} in {time.time()-t0:.0f}s (saved {args.ckpt})",
          flush=True)


if __name__ == "__main__":
    main()
