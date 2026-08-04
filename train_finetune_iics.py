"""Combine the two strongest levers: a fine-tuned DINOv2 backbone + the two-stage
IICS recipe (multi-branch per-camera classifiers + inter-camera classifier-score
similarity), trained longer on raw crops. Goal: push LABEL-FREE cross-view toward
the 0.85 supervised ceiling.

    python train_finetune_iics.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from cowreid import CameraTopology, Manifest, build_tracklets, extract_paths
from cowreid.cajaccard import dbscan_cluster, num_clusters
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.crossview_ot import embed_crops, mine_crop_ot_links
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, ReIDEvaluator, build_full_cross_camera
from cowreid.iics import MultiBranchReID, cluster_from_similarity, jaccard_from_scores
from cowreid.tracklets import TrackletIndex
from train_finetune import ClipLoader
from train_phase2 import build_objective
from train_phase2_run import make_masks, sample_frames

# best frozen / fine-tuned references so far (mAP, rank-1)
REF = {"leaveout": {"frozen_sup": (0.435, 0.393), "ft_sup": (0.863, 0.847),
                    "ft_ssl": (0.279, 0.472)},
       "full": {"ft_ssl": (0.232, 0.337)}}


class FineTuneIICS(nn.Module):
    def __init__(self, backbone: DinoV2Backbone, n_classes_per_cam, proj_dim=256):
        super().__init__()
        self.backbone = backbone
        self.head = MultiBranchReID(backbone.embed_dim, n_classes_per_cam, proj_dim)
        self.proj_dim = proj_dim

    def _frames(self, clips):
        B, T = clips.shape[:2]
        return self.backbone(clips.flatten(0, 1)).view(B, T, -1)

    def embed(self, clips):
        return self.head.backbone(self._frames(clips))

    def logits(self, emb, cam):
        return self.head.logits(emb, cam)

    def all_scores(self, emb):
        return self.head.all_scores(emb)


def crossview_candidates(index, topo, train_tids, overlap_thr=0.02):
    """Tracklet pairs eligible to be cross-view positives: overlapping camera pair +
    overlapping time interval (the free temporal-sync signal)."""
    ov = topo.overlapping_pairs(overlap_thr)
    tl = sorted((index[t] for t in train_tids), key=lambda x: x.t_start)
    cand = []
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            if tl[j].t_start > tl[i].t_end:
                break
            if tl[i].camera == tl[j].camera:
                continue
            if frozenset((tl[i].camera, tl[j].camera)) in ov and tl[i].overlaps_in_time(tl[j]):
                cand.append((tl[i].tracklet_id, tl[j].tracklet_id))
    return cand


def mine_must_links(emb, candidates, sim_thr, gt=None):
    """Accept a candidate as a cross-view positive if both ends are each other's best
    candidate partner (mutual-NN) and cosine >= sim_thr. Returns (links, gt_precision)."""
    if not candidates:
        return [], None
    scored = [(float(emb[a] @ emb[b]), a, b) for a, b in candidates]
    best = {}
    for s, a, b in scored:
        if s > best.get(a, (-2, None))[0]:
            best[a] = (s, b)
        if s > best.get(b, (-2, None))[0]:
            best[b] = (s, a)
    links = {frozenset((a, b)) for s, a, b in scored
             if s >= sim_thr and best[a][1] == b and best[b][1] == a}
    prec = None
    if gt is not None and links:
        prec = float(np.mean([gt[tuple(l)[0]] == gt[tuple(l)[1]] for l in links]))
    return list(links), prec


def merge_labels(labels, must_links):
    """Force-merge clusters connected by mined must-links (union-find)."""
    parent = {t: t for t in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    grp = defaultdict(list)
    for t, l in labels.items():
        grp[l].append(t)
    for members in grp.values():
        for t in members[1:]:
            union(members[0], t)
    for l in must_links:
        a, b = tuple(l)
        union(a, b)
    roots, out, nxt = {}, {}, 0
    for t in labels:
        r = find(t)
        if r not in roots:
            roots[r] = nxt; nxt += 1
        out[t] = roots[r]
    return out


def semisup_labels(train_tids, emb, cam_of, eps, cl, labeled_set, gt):
    """Semi-supervised labels: enrolled (labeled) tracklets get their GT identity
    (shared across cameras -> a correct cross-camera anchor); the rest are clustered
    (CA-Jaccard DBSCAN) and given ids offset after the labeled identities."""
    lab_gts = sorted({gt[t] for t in labeled_set})
    gtmap = {g: i for i, g in enumerate(lab_gts)}
    labels = {t: gtmap[gt[t]] for t in labeled_set}
    offset = len(lab_gts)
    unl = [t for t in train_tids if t not in labeled_set]
    if unl:
        from cowreid.cajaccard import dbscan_cluster as _dbscan
        feats = np.stack([emb[t] for t in unl])
        clab = _dbscan(unl, feats, [cam_of(t) for t in unl], eps=eps, cannot_link=cl)
        for t, l in clab.items():
            labels[t] = offset + l
    return labels


@torch.no_grad()
def embed_all(model, loader, tids, T, device, with_scores=False):
    model.eval()
    E, S = {}, {}
    for i in range(0, len(tids), 16):
        chunk = tids[i:i + 16]
        x = loader.batch(chunk, T, train=False)
        with torch.autocast("cuda", dtype=torch.float16):
            emb = model.embed(x)
            sc = model.all_scores(emb) if with_scores else None
        for k, t in enumerate(chunk):
            E[t] = emb[k].float().cpu().numpy()
            if with_scores:
                S[t] = sc[k].float().cpu().numpy()
    return (E, S) if with_scores else E


def inter_labels(model, loader, tids, T, device, n_cam, cl, mu=1.0, thr=0.5, k=10):
    E, S = embed_all(model, loader, tids, T, device, with_scores=True)
    Em = np.stack([E[t] for t in tids]); Sm = np.stack([S[t] for t in tids])
    sim = Em @ Em.T + mu * jaccard_from_scores(Sm, n_cam)
    return cluster_from_similarity(list(tids), sim, thr, k, cl)


def train(loader, train_tids, index, cl, device, steps, refresh_every, P, K, T,
          proj_dim, n_blocks, seed, frozen_mean, eps=0.6, topo=None, gt=None,
          crossview=True, sim_thr=0.8, mine_mode="tracklet", crop_bags=None,
          crop_paths=None, ot_min_conf=0.5, ot_min_votes=3, labeled_tids=None,
          model_name="vit_small_patch14_dinov2.lvd142m"):
    rng = np.random.default_rng(seed)
    cams = sorted({index.camera_of(t) for t in train_tids})
    train_cams = [index.camera_of(t) for t in train_tids]
    cv_pairs = crossview_candidates(index, topo, train_tids) if (crossview and topo) else []
    if crossview:
        print(f"    cross-view candidates: {len(cv_pairs)}", flush=True)

    # intra-camera pseudo-labels from frozen features (init), per-camera classifier sizes
    cl_same = {p for p in cl if len({index.camera_of(t) for t in p}) == 1}
    intra, n_cls = {}, {}
    by_cam = defaultdict(list)
    for t in train_tids:
        by_cam[index.camera_of(t)].append(t)
    for c, ts in by_cam.items():
        lab = ClusterAssigner(0.7, 10).assign(ts, np.stack([frozen_mean[t] for t in ts]), cl_same)
        intra.update(lab); n_cls[c] = ClusterAssigner.num_clusters(lab) or 1
    intra_pool = defaultdict(lambda: defaultdict(list))
    for t, l in intra.items():
        intra_pool[index.camera_of(t)][l].append(t)

    backbone = DinoV2Backbone(model_name=model_name, pretrained=True).requires_grad_(False)
    backbone.unfreeze_last(n_blocks)
    model = FineTuneIICS(backbone, n_cls, proj_dim).to(device)
    opt = torch.optim.AdamW(
        [{"params": backbone.trainable_parameters(), "lr": 1e-5},
         {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    ce = nn.CrossEntropyLoss()

    # inter-camera pseudo-labels (init on frozen features)
    if labeled_tids is not None:                     # semi-supervised
        inter = semisup_labels(train_tids, frozen_mean, index.camera_of, eps, cl,
                               labeled_tids, gt)
        print(f"    [init] semi-sup: {len(labeled_tids)} labeled tracklets, "
              f"{num_clusters(inter)} labels", flush=True)
    else:                                            # unsupervised CA-Jaccard + mining
        inter = dbscan_cluster(train_tids, np.stack([frozen_mean[x] for x in train_tids]),
                               train_cams, eps=eps, cannot_link=cl)
        if crossview:
            links, prec = mine_must_links(frozen_mean, cv_pairs, sim_thr, gt)
            inter = merge_labels(inter, links)
            print(f"    [init] must-links={len(links)} precision={prec}", flush=True)
    obj, mem = build_objective(proj_dim, max(1, num_clusters(inter)))
    obj.to(device)

    for step in range(steps):
        if step > 0 and step % refresh_every == 0:
            E = embed_all(model, loader, train_tids, T, device)
            if labeled_tids is not None:             # semi-supervised refresh
                inter = semisup_labels(train_tids, E, index.camera_of, eps, cl,
                                       labeled_tids, gt)
                mem.reset(max(1, num_clusters(inter)))
                continue_crossview = False
            else:
                inter = dbscan_cluster(train_tids, np.stack([E[t] for t in train_tids]),
                                       train_cams, eps=eps, cannot_link=cl)
                continue_crossview = crossview
            if continue_crossview:
                if mine_mode == "crop_ot" and crop_bags:
                    crop_emb = embed_crops(model, loader.loader, loader.tf, crop_paths, device)
                    links, prec, ncand = mine_crop_ot_links(
                        crop_bags, crop_emb, index.tracklet_of, min_conf=ot_min_conf,
                        min_votes=ot_min_votes, gt=gt)
                    print(f"    [refresh {step}] crop-OT links={len(links)}/{ncand} "
                          f"precision={prec}", flush=True)
                else:
                    links, prec = mine_must_links(E, cv_pairs, sim_thr, gt)
                    print(f"    [refresh {step}] must-links={len(links)} precision={prec}",
                          flush=True)
                inter = merge_labels(inter, links)
            mem.reset(max(1, num_clusters(inter)))

        if step % 2 == 0:                       # intra-camera multi-branch CE
            cam = cams[int(rng.integers(len(cams)))]
            pool = [l for l, ts in intra_pool[cam].items() if ts]
            if not pool:
                continue
            chosen = rng.choice(pool, size=min(P, len(pool)), replace=False)
            tids, labs = [], []
            for l in chosen:
                cand = intra_pool[cam][int(l)]
                pick = rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K)
                tids += pick.tolist(); labs += [int(l)] * len(pick)
            x = loader.batch(tids, T, train=True)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = ce(model.logits(model.embed(x), cam),
                          torch.tensor(labs, device=device))
        else:                                   # inter-camera cluster + topology
            pool = defaultdict(list)
            for t, l in inter.items():
                pool[l].append(t)
            chosen = rng.choice(list(pool), size=min(P, len(pool)), replace=False)
            tids = []
            for l in chosen:
                cand = pool[int(l)]
                tids += rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K).tolist()
            x = loader.batch(tids, T, train=True)
            labs, pos, hard, clp = make_masks(tids, inter, cl)
            model.train()
            with torch.autocast("cuda", dtype=torch.float16):
                emb = model.embed(x)
                loss, _ = obj(emb, positive_mask=pos.to(device),
                              hard_negative_mask=hard.to(device),
                              cluster_labels=labs.to(device),
                              cannot_link_pairs=clp.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 100 == 0:
            print(f"    step {step:4d} loss={float(loss.detach()):.3f} "
                  f"#inter={num_clusters(inter)}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.6, help="DBSCAN eps on CA-Jaccard distance")
    ap.add_argument("--crossview-sim", type=float, default=0.8,
                    help="cosine threshold for mined cross-view must-links")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    need = sorted({p for t in tracklets for p in sample_frames(t, args.frames)})
    extract_paths(args.tar, need, args.work)
    loader = ClipLoader(index, args.work, args.frames, args.image_size, device,
                        np.random.default_rng(args.seed))
    d = np.load(args.cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}
    ev = ReIDEvaluator(ranks=(1, 5, 10))

    def report(tag, q, g, model, refs):
        emb = embed_all(model, loader, sorted({it.tracklet_id for it in q + g}), args.frames, device)
        r = ev.evaluate(q, g, emb)
        print(f"\n>>> {tag}: fine-tuned IICS  mAP={r['mAP']:.3f} rank-1={r['rank-1']:.3f} "
              f"rank-5={r['rank-5']:.3f}", flush=True)
        for name, (m, r1) in refs.items():
            print(f"      vs {name}: mAP={m:.3f} rank-1={r1:.3f}  "
                  f"(Δ rank-1 {r['rank-1']-r1:+.3f})", flush=True)

    hold = args.holdout_camera
    print("\n========== LEAVE-OUT 66.130: fine-tuned + IICS ==========")
    gal_ids = {t.gt_label for t in tracklets if t.camera != hold}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != hold]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gal_ids]
    loco_train = [t.tracklet_id for t in tracklets if t.camera != hold]
    m1 = train(loader, loco_train, index, cl, device, args.steps, args.refresh_every,
               args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed, frozen_mean,
               eps=args.eps, topo=topo, gt=gt, crossview=True, sim_thr=args.crossview_sim)
    report("LEAVE-OUT 66.130", query, gallery, m1, REF["leaveout"])

    print("\n========== FULL transductive: fine-tuned + IICS ==========")
    fq, fg = build_full_cross_camera(tracklets)
    full_tids = sorted({it.tracklet_id for it in fq})
    m2 = train(loader, full_tids, index, cl, device, args.steps, args.refresh_every,
               args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed, frozen_mean,
               eps=args.eps, topo=topo, gt=gt, crossview=True, sim_thr=args.crossview_sim)
    report("FULL", fq, fg, m2, REF["full"])


if __name__ == "__main__":
    main()
