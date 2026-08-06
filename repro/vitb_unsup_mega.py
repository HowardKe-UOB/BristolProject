"""MegaDescriptor student: the proven deployment-mode distillation recipe with the
backbone swapped to MegaDescriptor-L-384 (Swin-L, animal-ReID foundation model,
`hf-hub:BVRA/MegaDescriptor-L-384`). Frozen features already beat frozen DINOv2 on
the bottleneck (dorsal 0.50 vs 0.42, P2 0.46 vs 0.34) -> test whether training it
with our recipe beats the DINOv2 students (P2 0.585).

Teacher = k=2 trio space over ALL 7 cameras (deployment mode, same as
_vitb_dep_s10..12). Inputs resized 518->384 on the fly. Only the last Swin stage
+ final norm are fine-tuned (lightweight).

Train:  python vitb_unsup_mega.py --seed 40 --ckpt _vitb_mega_s40_ckpt.pt --wall 300 --target 1000
Eval:   python eval_sweep.py --ckpts _vitb_mega_s40_ckpt.pt --tag mega_s40 --backbone mega
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
import timm

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.tracklets import TrackletIndex
from consensus_ens import mutual_knn_links
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader, embed_tids

MEGA = "hf-hub:BVRA/MegaDescriptor-L-384"
MEGA_IMG = 384
K2 = ("s7", "s8", "s9")


class MegaBackbone(nn.Module):
    """MegaDescriptor Swin-L/384. Fine-tunes the last `n_stage` stages + norm."""

    def __init__(self, model_name=MEGA, pretrained=True, n_stage=1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.model.num_features
        for p in self.model.parameters():
            p.requires_grad_(False)
        stages = self.model.layers
        for st in list(stages)[-n_stage:]:
            for p in st.parameters():
                p.requires_grad_(True)
        if hasattr(self.model, "norm"):
            for p in self.model.norm.parameters():
                p.requires_grad_(True)
        self.frozen = False

    def forward(self, x):
        return self.model(x)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]


class MegaStudent(FineTuneIICS):
    def _frames(self, clips):
        B, T = clips.shape[:2]
        x = clips.flatten(0, 1)
        if x.shape[-1] != MEGA_IMG:
            x = F.interpolate(x, size=(MEGA_IMG, MEGA_IMG), mode="bilinear",
                              align_corners=False)
        return self.backbone(x).view(B, T, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--wall", type=float, default=300)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-stage", type=int, default=1)
    ap.add_argument("--link-k", type=int, default=2)
    ap.add_argument("--w-link", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--momentum", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--ckpt", default="_vitb_mega_s40_ckpt.pt")
    ap.add_argument("--teacher-npz", default="_vitb_dst_emb_v4.npz")
    ap.add_argument("--supervised", action="store_true",
                    help="use GT identities as labels (fair supervised baseline)")
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

    d = np.load(args.teacher_npz, allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    tkeys = [s for s in sorted(d.files) if s != "ids" and (any(k in s for k in K2) or s == "t0")]
    Xt = np.mean([d[s] for s in tkeys], axis=0)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12)
    print(f"[teacher-space] {args.teacher_npz} keys={tkeys} dim={Xt.shape[1]}", flush=True)

    g_tids = list(ids)                                  # all 7 cameras (deployment)
    cams_list = [cam_of(t) for t in g_tids]
    cams = sorted(set(cams_list))

    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cl_same = {p for p in cl if len({cam_of(t) for t in p}) == 1}
    by_cam = defaultdict(list)
    for t in g_tids:
        by_cam[cam_of(t)].append(t)
    intra_global, n_cls, intra_local = {}, {}, {}
    off = 0
    for c, ts in sorted(by_cam.items()):
        if args.supervised:                             # GT identity = intra label
            gids = sorted({gt[t] for t in ts}); gm = {g: i for i, g in enumerate(gids)}
            lab = {t: gm[gt[t]] for t in ts}
        else:
            E = np.stack([Xt[pos[t]] for t in ts])
            lab = ClusterAssigner(0.7, 10).assign(ts, E, cl_same)
        for t in ts:
            intra_local[t] = lab[t]; intra_global[t] = off + lab[t]
        k = (ClusterAssigner.num_clusters(lab) if not args.supervised
             else len({lab[t] for t in ts})) or 1
        n_cls[c] = k; off += k
    n_proxy = off

    plinks = defaultdict(list)
    if args.supervised:                                 # all same-GT cross-cam proxy pairs
        by_gid = defaultdict(list)
        for t in g_tids:
            by_gid[gt[t]].append(intra_global[t])
        for gid, pxs in by_gid.items():
            uniq = sorted(set(pxs))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    plinks[uniq[i]].append((uniq[j], 1.0))
                    plinks[uniq[j]].append((uniq[i], 1.0))
        n_links = sum(len(v) for v in plinks.values()) // 2
    else:
        X = np.stack([Xt[pos[t]] for t in g_tids]).astype(np.float32)
        links = mutual_knn_links(X.copy(), cams_list, k=args.link_k)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        for a, b in (tuple(l) for l in links):
            ca, cb = intra_global[g_tids[a]], intra_global[g_tids[b]]
            conf = float(Xn[a] @ Xn[b])
            plinks[ca].append((cb, conf)); plinks[cb].append((ca, conf))
        n_links = len(links)
    print(f"[{'SUPERVISED' if args.supervised else 'teacher'}] {n_proxy} proxies, "
          f"{n_links} links", flush=True)

    proxy_members = defaultdict(list); proxy_cam = {}
    for t in g_tids:
        proxy_members[intra_global[t]].append(t)
        proxy_cam[intra_global[t]] = cam_of(t)
    pcam_arr = np.array([proxy_cam[i] for i in range(n_proxy)])
    linked_pool = sorted(plinks)
    intra_pool = defaultdict(lambda: defaultdict(list))
    for t in g_tids:
        intra_pool[cam_of(t)][intra_local[t]].append(t)

    backbone = MegaBackbone(n_stage=args.n_stage)
    model = MegaStudent(backbone, n_cls, args.proj_dim).to(device)
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

    step = start_step
    while step < args.target and (time.time() - t0) < args.wall:
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
            sim = embf @ proxies.clone().t() / args.temp
            l_items = []
            for i in range(len(tids)):
                p = pidx[i]; c = proxy_cam[p]
                sc = same_cam_idx[c]
                tgt = int(np.where(sc == p)[0][0])
                l_i = F.cross_entropy(sim[i, sc].unsqueeze(0),
                                      torch.tensor([tgt], device=device))
                if p in plinks:
                    dc = diff_cam_idx[c]
                    den = torch.logsumexp(sim[i, dc], dim=0)
                    l_link = 0.0; wsum = 0.0
                    for (op, conf) in plinks[p]:
                        l_link = l_link + conf * (den - sim[i, op])
                        wsum += conf
                    l_i = l_i + args.w_link * l_link / max(wsum, 1e-9)
                l_items.append(l_i)
            loss = torch.stack(l_items).mean()
            with torch.no_grad():
                for f, p in zip(embf.detach(), pidx):
                    proxies[p] = F.normalize(args.momentum * proxies[p]
                                             + (1 - args.momentum) * f, dim=0)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        step += 1
        if step % 100 == 0:
            print(f"  step {step} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, args.ckpt)
    print(f"[chunk] -> step {step}/{args.target} in {time.time()-t0:.0f}s "
          f"(saved {args.ckpt})", flush=True)


if __name__ == "__main__":
    main()
