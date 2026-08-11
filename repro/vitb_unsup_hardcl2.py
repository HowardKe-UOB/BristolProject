"""DEPLOYMENT-MODE student + HARD CANNOT-LINK v2 (denser ammo, in-softmax negatives).

Combines the two proven P2 levers for the first time:
  * deployment mode (all 7 cameras, links mined across all pairs) -- the P2
    winner (trio 0.549 vs holdout 0.516);
  * hard cannot-link sharpening (dorsal +1.9~3.6 in holdout-mode ablations),
    upgraded: (a) same-camera pairs mined down to cos>=0.20 (100% clean by
    construction -> thousands of pairs instead of hundreds), (b) hard partners
    injected as BOOSTED NEGATIVES inside both softmax losses (dense gradient
    every step) in addition to the sampled embedding hinge.

Teacher = holdout k=2 trio space (same as the deploy trio), so comparisons vs
`_vitb_dep_s10..12` isolate the hardCL-v2 additions.

Train:  python vitb_unsup_hardcl2.py --seed 16 --ckpt _vitb_hc2_s16_ckpt.pt --wall 240 --target 1000
Eval:   python eval_sweep.py --ckpts _vitb_hc2_s16_ckpt.pt --tag hc2_s16
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

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
from vitb_unsup import CACHE_JSON, CACHE_NPY, VITB, CacheLoader, embed_tids

K2 = ("s7", "s8", "s9")


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
    ap.add_argument("--link-k", type=int, default=2)
    ap.add_argument("--w-link", type=float, default=1.0)
    ap.add_argument("--hcl-same-thr", type=float, default=0.20)
    ap.add_argument("--hcl-cross-thr", type=float, default=0.40)
    ap.add_argument("--hcl-margin", type=float, default=0.20)
    ap.add_argument("--w-hcl", type=float, default=0.5)
    ap.add_argument("--hcl-pairs-per-step", type=int, default=4)
    ap.add_argument("--neg-boost", type=float, default=0.15,
                    help="cosine-units boost added to hard-negative logits")
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--momentum", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--ckpt", default="_vitb_hc2_s16_ckpt.pt")
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

    # ---------- teacher: k=2 trio space over ALL cameras (deploy mode) ---------- #
    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    Xt = np.mean([d[s] for s in sorted(d.files) if any(k in s for k in K2)], axis=0)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)
    have = set(ids)

    g_tids = list(ids)                                  # all 7 cameras
    cams_list = [cam_of(t) for t in g_tids]
    cams = sorted(set(cams_list))

    cl_same_cam = {p for p in cl if len({cam_of(t) for t in p}) == 1}
    by_cam = defaultdict(list)
    for t in g_tids:
        by_cam[cam_of(t)].append(t)
    intra_global, n_cls, intra_local = {}, {}, {}
    off = 0
    for c, ts in sorted(by_cam.items()):
        E = np.stack([Xt[pos[t]] for t in ts])
        lab = ClusterAssigner(0.7, 10).assign(ts, E, cl_same_cam)
        for t in ts:
            intra_local[t] = lab[t]; intra_global[t] = off + lab[t]
        k = ClusterAssigner.num_clusters(lab) or 1
        n_cls[c] = k; off += k
    n_proxy = off

    X = np.stack([Xt[pos[t]] for t in g_tids]).astype(np.float32)
    links = mutual_knn_links(X.copy(), cams_list, k=args.link_k)
    plinks = defaultdict(list)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    for a, b in (tuple(l) for l in links):
        ca, cb = intra_global[g_tids[a]], intra_global[g_tids[b]]
        conf = float(Xn[a] @ Xn[b])
        plinks[ca].append((cb, conf)); plinks[cb].append((ca, conf))
    print(f"[teacher] {n_proxy} proxies (7 cams), {len(links)} mutual-NN links",
          flush=True)

    # ---------- hard cannot-link pairs v2 (denser, all cameras) ---------- #
    hcl_pairs = []
    hard_prox = defaultdict(set)                        # proxy -> hard partner proxies
    hard_intra = defaultdict(lambda: defaultdict(set))  # cam -> local class -> classes
    for p in cl:
        a, b = tuple(p)
        if a not in have or b not in have:
            continue
        s = float(Xt[pos[a]] @ Xt[pos[b]])
        same = cam_of(a) == cam_of(b)
        thr = args.hcl_same_thr if same else args.hcl_cross_thr
        if s < thr:
            continue
        hcl_pairs.append((a, b, s))
        pa, pb = intra_global[a], intra_global[b]
        if pa != pb:
            hard_prox[pa].add(pb); hard_prox[pb].add(pa)
        if same:
            c = cam_of(a); la, lb = intra_local[a], intra_local[b]
            if la != lb:
                hard_intra[c][la].add(lb); hard_intra[c][lb].add(la)
    n_same = sum(1 for a, b, _ in hcl_pairs if cam_of(a) == cam_of(b))
    print(f"[hardCL v2] {len(hcl_pairs)} pairs ({n_same} same-cam @thr "
          f"{args.hcl_same_thr}, {len(hcl_pairs)-n_same} cross-cam @thr "
          f"{args.hcl_cross_thr}); {len(hard_prox)} proxies with hard partners",
          flush=True)

    proxy_members = defaultdict(list); proxy_cam = {}
    for t in g_tids:
        proxy_members[intra_global[t]].append(t)
        proxy_cam[intra_global[t]] = cam_of(t)
    pcam_arr = np.array([proxy_cam[i] for i in range(n_proxy)])
    linked_pool = sorted(plinks)
    intra_pool = defaultdict(lambda: defaultdict(list))
    for t in g_tids:
        intra_pool[cam_of(t)][intra_local[t]].append(t)

    backbone = DinoV2Backbone(model_name=VITB, pretrained=True).requires_grad_(False)
    backbone.unfreeze_last(args.n_blocks)
    model = FineTuneIICS(backbone, n_cls, args.proj_dim).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
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

    E0 = embed_tids(model, cload, g_tids, args.T, device)
    P0 = np.stack([np.mean([E0[t] for t in proxy_members[i]], axis=0) for i in range(n_proxy)])
    P0 = P0 / (np.linalg.norm(P0, axis=1, keepdims=True) + 1e-12)
    proxies = F.normalize(torch.tensor(P0, dtype=torch.float32, device=device), dim=1)
    same_cam_idx = {c: np.where(pcam_arr == c)[0] for c in cams}
    diff_cam_idx = {c: np.where(pcam_arr != c)[0] for c in cams}
    boost_ce = args.neg_boost * 16.0                    # cosine classifier scale=16
    boost_px = args.neg_boost / args.temp

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        npair = min(args.hcl_pairs_per_step, len(hcl_pairs))
        sel = rng.choice(len(hcl_pairs), size=npair, replace=False) if npair else []
        hcl_tids = []
        for si in sel:
            a, b, _s = hcl_pairs[int(si)]
            hcl_tids += [a, b]

        if step % 2 == 0:                              # intra CE (+boosted hard negs)
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
                emb = model.embed(cload.batch(tids + hcl_tids, args.T))
            embf = emb.float()
            logits = model.logits(embf[:len(tids)], cam).float()
            for i, l in enumerate(labs):                # boost hard same-cam negatives
                for l2 in hard_intra[cam].get(l, ()):
                    logits[i, l2] = logits[i, l2] + boost_ce
            loss = F.cross_entropy(logits, torch.tensor(labs, device=device))
        else:                                          # proxy CE + links (+boosted negs)
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
                emb = model.embed(cload.batch(tids + hcl_tids, args.T))
            embf = emb.float()
            sim = embf[:len(tids)] @ proxies.clone().t() / args.temp
            l_items = []
            for i in range(len(tids)):
                p = pidx[i]; c = proxy_cam[p]
                row = sim[i]
                if p in hard_prox:                     # dense in-softmax hard negatives
                    hp = list(hard_prox[p])
                    row = row.clone()
                    row[hp] = row[hp] + boost_px
                sc = same_cam_idx[c]
                tgt = int(np.where(sc == p)[0][0])
                l_i = F.cross_entropy(row[sc].unsqueeze(0),
                                      torch.tensor([tgt], device=device))
                if p in plinks:
                    dc = diff_cam_idx[c]
                    den = torch.logsumexp(row[dc], dim=0)
                    l_link = 0.0; wsum = 0.0
                    for (op, conf) in plinks[p]:
                        l_link = l_link + conf * (den - row[op])
                        wsum += conf
                    l_i = l_i + args.w_link * l_link / max(wsum, 1e-9)
                l_items.append(l_i)
            loss = torch.stack(l_items).mean()
            with torch.no_grad():
                for f, p in zip(embf[:len(tids)].detach(), pidx):
                    proxies[p] = F.normalize(args.momentum * proxies[p]
                                             + (1 - args.momentum) * f, dim=0)
        if npair:                                      # embedding hinge (both branches)
            h = embf[-2 * npair:]
            l_hcl = 0.0
            for k in range(npair):
                cos = (h[2 * k] * h[2 * k + 1]).sum()
                l_hcl = l_hcl + F.relu(cos - args.hcl_margin)
            loss = loss + args.w_hcl * l_hcl / npair
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1
        if step % 200 == 0:
            print(f"  step {step} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, args.ckpt)
    print(f"[chunk] -> step {step}/{args.target} in {time.time()-t0:.0f}s "
          f"(saved {args.ckpt})", flush=True)


if __name__ == "__main__":
    main()
