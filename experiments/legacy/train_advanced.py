"""Consolidated comparison: frozen DINOv2 vs SSL head vs IICS vs SUPERVISED upper
bound, on both the FULL (transductive) and leave-out-66.130 (inductive) protocols.

All methods train a small head over the SAME cached frozen DINOv2 features, so the
comparison is apples-to-apples and isolates the effect of the label/similarity recipe.

    python train_advanced.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
from collections import defaultdict

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.eval import EvalItem, ReIDEvaluator, build_full_cross_camera
from cowreid.iics import MultiBranchReID, inter_camera_labels
from cowreid.tracklets import TrackletIndex
from train_phase2 import build_objective
from train_phase2_run import (cache_features, frozen_embeddings, head_embeddings,
                              make_masks, train_head)


def contiguous(label_map):
    uniq = {v: i for i, v in enumerate(sorted(set(label_map.values())))}
    return {k: uniq[v] for k, v in label_map.items()}


def same_camera_cl(cl, index):
    return {p for p in cl if len({index.camera_of(t) for t in p}) == 1}


# --------------------------------------------------------------------------- #
def train_iics(clips, train_tids, index, cl, device, steps, refresh_every,
               P, K, T, proj_dim, seed):
    rng = np.random.default_rng(seed)
    in_dim = clips[train_tids[0]].shape[1]
    cams = sorted({index.camera_of(t) for t in train_tids})
    tids_by_cam = defaultdict(list)
    for t in train_tids:
        tids_by_cam[index.camera_of(t)].append(t)

    # intra-camera pseudo-labels (within-camera clustering on frozen feats)
    intra, n_cls = {}, {}
    cl_same = same_camera_cl(cl, index)
    for c, ts in tids_by_cam.items():
        feats = np.stack([frozen_embeddings(clips, [t])[t] for t in ts])
        lab = ClusterAssigner(sim_threshold=0.7, k=10).assign(ts, feats, cl_same)
        intra.update({t: lab[t] for t in ts})
        n_cls[c] = ClusterAssigner.num_clusters(lab) or 1
    intra_by_cam_label = defaultdict(lambda: defaultdict(list))
    for t, l in intra.items():
        intra_by_cam_label[index.camera_of(t)][l].append(t)

    model = MultiBranchReID(in_dim, n_cls, proj_dim=proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    inter = inter_camera_labels(model, clips, train_tids, device, len(cams),
                                cannot_link=cl)
    objective, memory = build_objective(proj_dim, max(1, ClusterAssigner.num_clusters(inter)))
    objective.to(device)
    inter_by_label = defaultdict(list)

    def clip_batch(tids):
        arr = np.stack([clips[t][rng.integers(0, len(clips[t]), size=T)] for t in tids])
        return torch.tensor(arr, dtype=torch.float32, device=device)

    for step in range(steps):
        if step > 0 and step % refresh_every == 0:
            inter = inter_camera_labels(model, clips, train_tids, device, len(cams),
                                        cannot_link=cl)
            memory.reset(max(1, ClusterAssigner.num_clusters(inter)))

        if step % 2 == 0:   # ---- intra-camera (multi-branch CE) ----
            cam = cams[int(rng.integers(len(cams)))]
            pool = [l for l, ts in intra_by_cam_label[cam].items() if ts]
            if not pool:
                continue
            chosen = rng.choice(pool, size=min(P, len(pool)), replace=False)
            tids, labs = [], []
            for l in chosen:
                cand = intra_by_cam_label[cam][int(l)]
                pick = rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K)
                tids += pick.tolist(); labs += [int(l)] * len(pick)
            emb = model.backbone(clip_batch(tids))
            loss = ce(model.logits(emb, cam), torch.tensor(labs, device=device))
        else:               # ---- inter-camera (cluster + topology constraints) ----
            inter_by_label.clear()
            for t, l in inter.items():
                inter_by_label[l].append(t)
            pool = list(inter_by_label)
            chosen = rng.choice(pool, size=min(P, len(pool)), replace=False)
            tids = []
            for l in chosen:
                cand = inter_by_label[int(l)]
                tids += rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K).tolist()
            emb = model.backbone(clip_batch(tids))
            labs, pos, hard, clp = make_masks(tids, inter, cl)
            loss, _ = objective(emb, positive_mask=pos.to(device),
                                hard_negative_mask=hard.to(device),
                                cluster_labels=labs.to(device),
                                cannot_link_pairs=clp.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"    iics step {step:4d}  loss={float(loss.detach()):.3f}  "
                  f"#inter_clusters={ClusterAssigner.num_clusters(inter)}")

    def embed_all(tids):
        model.eval(); out = {}
        with torch.no_grad():
            for t in tids:
                x = torch.tensor(clips[t][None], dtype=torch.float32, device=device)
                out[t] = model.backbone(x)[0].cpu().numpy()
        return out
    return embed_all


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--refresh-every", type=int, default=200)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    cl = build_cannot_link(tracklets, topo, 0.02)
    clips = cache_features(tracklets, args.tar, args.work, args.cache, args.frames, device)
    ev = ReIDEvaluator(ranks=(1, 5, 10))
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    def run(label, q, g, train_tids):
        rows = {}
        eval_tids = sorted({it.tracklet_id for it in q + g})
        rows["frozen"] = ev.evaluate(q, g, frozen_embeddings(clips, eval_tids))
        # SSL
        enc = train_head(clips, train_tids, cl, device, args.steps, args.refresh_every,
                         args.P, args.K, args.T, args.proj_dim, args.seed)
        rows["ssl"] = ev.evaluate(q, g, head_embeddings(enc, clips, eval_tids, device))
        # IICS
        embed_iics = train_iics(clips, train_tids, index, cl, device, args.steps,
                                args.refresh_every, args.P, args.K, args.T, args.proj_dim, args.seed)
        rows["iics"] = ev.evaluate(q, g, embed_iics(eval_tids))
        # SUPERVISED upper bound (GT labels)
        sup_labels = contiguous({t: gt[t] for t in train_tids})
        enc_s = train_head(clips, train_tids, cl, device, args.steps, args.refresh_every,
                           args.P, args.K, args.T, args.proj_dim, args.seed,
                           fixed_labels=sup_labels)
        rows["supervised"] = ev.evaluate(q, g, head_embeddings(enc_s, clips, eval_tids, device))
        print(f"\n===== {label} =====")
        for m, r in rows.items():
            print(f"  {m:11s} mAP={r['mAP']:.3f}  rank-1={r['rank-1']:.3f}  "
                  f"rank-5={r['rank-5']:.3f}  rank-10={r['rank-10']:.3f}")
        return rows

    # FULL transductive
    fq, fg = build_full_cross_camera(tracklets)
    run("FULL", fq, fg, sorted({it.tracklet_id for it in fq}))

    # LEAVE-OUT 66.130 inductive
    hold = args.holdout_camera
    gal_ids = {t.gt_label for t in tracklets if t.camera != hold}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
               for t in tracklets if t.camera != hold]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gal_ids]
    train_tids = [t.tracklet_id for t in tracklets if t.camera != hold]
    run(f"LEAVE-OUT {hold}", query, gallery, train_tids)


if __name__ == "__main__":
    main()
