"""ACTIVE PAIRWISE LABELING simulation: the click-curve experiment.

Protocol: the system proposes its N most-confused cross-camera tracklet pairs
(top ensemble similarity, excluding physics-decided cannot-links -- selection is
LABEL-FREE); a simulated human (ground truth standing in for a person, exactly
equivalent) answers same/different. Answers are used three ways:
  yes  -> verified positive link (confidence 1.0, pairwise loss);
  no   -> verified hard negative (hinge + in-softmax boost);
  no on a previously MINED link -> that false link is removed (cleaning).
Everything else matches the deployment student (all 7 cameras, trio teacher,
frozen labels, 1000 steps): the N=0 baseline is `_vitb_dep_s10_ckpt.pt`
(P2 0.521 / dorsal 0.511 / P1t 0.902 single-model).

Train:  python vitb_unsup_active.py --clicks 300 --seed 31 --ckpt _vitb_act_n300_ckpt.pt --wall 480 --target 1000
Eval:   python eval_sweep.py --ckpts _vitb_act_n300_ckpt.pt --tag act_n300
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
    ap.add_argument("--clicks", type=int, required=True)
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=480)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--link-k", type=int, default=2)
    ap.add_argument("--w-link", type=float, default=1.0)
    ap.add_argument("--hcl-margin", type=float, default=0.20)
    ap.add_argument("--w-hcl", type=float, default=0.5)
    ap.add_argument("--hcl-pairs-per-step", type=int, default=4)
    ap.add_argument("--neg-boost", type=float, default=0.15)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--momentum", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    device = "cuda"; t0 = time.time()

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

    # ---------- teacher: k=2 trio space over ALL cameras ---------- #
    d = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    Xt = np.mean([d[s] for s in sorted(d.files) if any(k in s for k in K2)], axis=0)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)

    g_tids = list(ids)
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
    mined = mutual_knn_links(X.copy(), cams_list, k=args.link_k)
    mined_pairs = {frozenset((g_tids[a], g_tids[b])) for a, b in (tuple(l) for l in mined)}

    # ---------- ACTIVE SELECTION (label-free) + simulated clicks ---------- #
    S = X @ X.T
    cam_arr = np.array(cams_list)
    same_cam_mask = cam_arr[:, None] == cam_arr[None, :]
    S[same_cam_mask] = -2                              # cross-camera pairs only
    np.fill_diagonal(S, -2)
    cl_set = {frozenset(p) for p in cl}
    iu = np.triu_indices(len(g_tids), k=1)
    order = np.argsort(-S[iu])
    asked, yes_pairs, no_pairs = [], [], []
    for k in order:
        if len(asked) >= args.clicks:
            break
        i, j = iu[0][k], iu[1][k]
        if S[i, j] <= -1:
            continue
        pair = frozenset((g_tids[i], g_tids[j]))
        if pair in cl_set:                             # physics already knows: no click wasted
            continue
        asked.append(pair)
        (yes_pairs if gt[g_tids[i]] == gt[g_tids[j]] else no_pairs).append(pair)
    print(f"[active] asked {len(asked)} clicks -> YES {len(yes_pairs)} / "
          f"NO {len(no_pairs)}  (click precision {len(yes_pairs)/max(len(asked),1):.2f})",
          flush=True)

    asked_set = set(asked)
    kept_mined = mined_pairs - asked_set               # unasked mined links stay as-is
    cleaned = mined_pairs & set(no_pairs)              # false mined links removed
    print(f"[active] mined links: {len(mined_pairs)} -> kept {len(kept_mined)} "
          f"(cleaned {len(cleaned)} false, {len(mined_pairs & set(yes_pairs))} "
          f"verified true)", flush=True)

    # proxy-level structures
    plinks = defaultdict(list)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    tid_i = {t: i for i, t in enumerate(g_tids)}

    def add_link(a, b, conf):
        pa, pb = intra_global[a], intra_global[b]
        if pa != pb:
            plinks[pa].append((pb, conf)); plinks[pb].append((pa, conf))

    for p in kept_mined:
        a, b = tuple(p)
        add_link(a, b, float(Xn[tid_i[a]] @ Xn[tid_i[b]]))
    for p in yes_pairs:
        a, b = tuple(p)
        add_link(a, b, 1.0)                            # human-verified positives

    hcl_pairs = [(*tuple(p), 1.0) for p in no_pairs]   # human-verified negatives
    hard_prox = defaultdict(set)
    for a, b, _c in hcl_pairs:
        pa, pb = intra_global[a], intra_global[b]
        if pa != pb:
            hard_prox[pa].add(pb); hard_prox[pb].add(pa)
    print(f"[teacher] {n_proxy} proxies, {sum(len(v) for v in plinks.values())//2} "
          f"link entries, {len(hcl_pairs)} verified negatives", flush=True)

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

    E0 = embed_tids(model, cload, g_tids, args.T, device)
    P0 = np.stack([np.mean([E0[t] for t in proxy_members[i]], axis=0) for i in range(n_proxy)])
    P0 = P0 / (np.linalg.norm(P0, axis=1, keepdims=True) + 1e-12)
    proxies = F.normalize(torch.tensor(P0, dtype=torch.float32, device=device), dim=1)
    same_cam_idx = {c: np.where(pcam_arr == c)[0] for c in cams}
    diff_cam_idx = {c: np.where(pcam_arr != c)[0] for c in cams}
    boost_px = args.neg_boost / args.temp

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
        npair = min(args.hcl_pairs_per_step, len(hcl_pairs))
        sel = rng.choice(len(hcl_pairs), size=npair, replace=False) if npair else []
        hcl_tids = []
        for si in sel:
            a, b, _c = hcl_pairs[int(si)]
            hcl_tids += [a, b]

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
                emb = model.embed(cload.batch(tids + hcl_tids, args.T))
            embf = emb.float()
            loss = ce(model.logits(embf[:len(tids)], cam).float(),
                      torch.tensor(labs, device=device))
        else:
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
                if p in hard_prox:
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
        if npair:
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
