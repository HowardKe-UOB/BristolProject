"""Fine-tune the DINOv2 backbone (partial unfreeze) through raw crops, and measure
whether it raises accuracy / the ceiling vs the frozen-backbone results.

Runs three settings and prints new vs frozen-backbone numbers:
  * LEAVE-OUT 66.130 SUPERVISED  -> new cross-view *ceiling* (frozen was mAP .435/r1 .393)
  * LEAVE-OUT 66.130 SSL          -> does the method improve cross-view (frozen .124/.178)
  * FULL SSL (transductive)       -> method improvement on seen cameras (frozen .127/.171)

Crops are read from the directory populated earlier (8 frames/tracklet). Uses AMP;
backbone last-N blocks + norm are trainable, head trains at a higher LR.

    python train_finetune.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from cowreid import CameraTopology, ImageLoader, Manifest, build_tracklets, extract_paths
from cowreid.batch import build_transform
from cowreid.cluster import ClusterAssigner, build_cannot_link
from cowreid.encoder import DinoV2Backbone, VideoReIDEncoder
from cowreid.eval import EvalItem, ReIDEvaluator, build_full_cross_camera
from cowreid.tracklets import TrackletIndex
from train_phase2 import build_objective
from train_phase2_run import make_masks, sample_frames

FROZEN = {  # reference numbers from train_advanced.py (frozen backbone)
    "leaveout_supervised": (0.435, 0.393), "leaveout_ssl": (0.124, 0.178),
    "leaveout_iics": (0.146, 0.233), "leaveout_frozen": (0.117, 0.252),
    "full_ssl": (0.127, 0.171), "full_frozen": (0.082, 0.155),
}


def contiguous(d):
    u = {v: i for i, v in enumerate(sorted(set(d.values())))}
    return {k: u[v] for k, v in d.items()}


class ClipLoader:
    """Loads (B, T, 3, H, W) clips for tracklets from extracted crops."""

    def __init__(self, index, root, frames, image_size, device, rng):
        self.index = index
        self.loader = ImageLoader(root=root)
        self.tf = build_transform(image_size)
        self.frames = frames
        self.device = device
        self.rng = rng
        self.paths = {t.tracklet_id: sample_frames(t, frames) for t in index.tracklets}

    def clip(self, tid, T, train):
        ps = self.paths[tid]
        n = len(ps)
        if train:
            idx = self.rng.integers(0, n, size=T)
        else:
            idx = np.linspace(0, n - 1, min(T, n)).astype(int)
            if len(idx) < T:                       # pad short tracklets to T frames
                idx = np.concatenate([idx, np.full(T - len(idx), idx[-1])])
        return torch.stack([self.tf(self.loader.load(ps[i])) for i in idx])

    def batch(self, tids, T, train=True):
        return torch.stack([self.clip(t, T, train) for t in tids]).to(self.device)


@torch.no_grad()
def embed_all(enc, loader, tids, T, device):
    enc.eval()
    out = {}
    for i in range(0, len(tids), 16):
        chunk = tids[i:i + 16]
        x = loader.batch(chunk, T, train=False)
        with torch.autocast("cuda", dtype=torch.float16):
            e = enc(x)["embed"]
        for t, v in zip(chunk, e.float().cpu().numpy()):
            out[t] = v
    return out


def finetune(loader, train_tids, cl, device, steps, refresh_every, P, K, T,
             proj_dim, n_blocks, seed, mode, gt=None, init_labels=None):
    rng = np.random.default_rng(seed)
    enc = VideoReIDEncoder(DinoV2Backbone(pretrained=True).requires_grad_(False),
                           proj_dim=proj_dim, pool="attn")
    enc.backbone.unfreeze_last(n_blocks)
    enc.to(device)
    opt = torch.optim.AdamW(
        [{"params": enc.backbone.trainable_parameters(), "lr": 1e-5},
         {"params": list(enc.pool.parameters()) + list(enc.proj.parameters()), "lr": 3e-4}],
        weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    supervised = mode == "supervised"
    if supervised:
        labels = contiguous({t: gt[t] for t in train_tids})
    else:
        labels = dict(init_labels)  # from cached frozen features
    obj, mem = build_objective(proj_dim, max(1, ClusterAssigner.num_clusters(labels)))
    obj.to(device)

    for step in range(steps):
        if not supervised and step > 0 and step % refresh_every == 0:
            emb = embed_all(enc, loader, train_tids, T, device)
            labels = ClusterAssigner(0.6, 10).assign(
                train_tids, np.stack([emb[t] for t in train_tids]), cl)
            mem.reset(max(1, ClusterAssigner.num_clusters(labels)))

        by = defaultdict(list)
        for t, l in labels.items():
            if l >= 0:
                by[l].append(t)
        chosen = rng.choice(list(by), size=min(P, len(by)), replace=False)
        tids = []
        for l in chosen:
            cand = by[int(l)]
            tids += rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K).tolist()

        x = loader.batch(tids, T, train=True)
        labs, pos, hard, clp = make_masks(tids, labels, cl)
        enc.train()
        with torch.autocast("cuda", dtype=torch.float16):
            emb = enc(x)["embed"]
            loss, comp = obj(emb, positive_mask=pos.to(device),
                             hard_negative_mask=hard.to(device),
                             cluster_labels=labs.to(device),
                             cannot_link_pairs=clp.to(device))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 100 == 0:
            print(f"    [{mode}] step {step:4d} loss={float(comp['total']):.3f} "
                  f"#clusters={ClusterAssigner.num_clusters(labels)}", flush=True)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--refresh-every", type=int, default=150)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
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

    # ensure crops present
    need = sorted({p for t in tracklets for p in sample_frames(t, args.frames)})
    extract_paths(args.tar, need, args.work)
    loader = ClipLoader(index, args.work, args.frames, args.image_size, device,
                        np.random.default_rng(args.seed))

    # init labels for SSL from cached frozen features
    d = np.load(args.cache, allow_pickle=True)
    frozen_clips = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (frozen_clips[t].mean(0) /
                       (np.linalg.norm(frozen_clips[t].mean(0)) + 1e-12))
                   for t in frozen_clips}

    ev = ReIDEvaluator(ranks=(1, 5, 10))

    def show(tag, q, g, enc, key):
        emb = embed_all(enc, loader, sorted({it.tracklet_id for it in q + g}), args.frames, device)
        r = ev.evaluate(q, g, emb)
        fm, fr = FROZEN[key]
        print(f"\n>>> {tag}\n    fine-tuned: mAP={r['mAP']:.3f} rank-1={r['rank-1']:.3f} "
              f"rank-5={r['rank-5']:.3f}\n    frozen ref: mAP={fm:.3f} rank-1={fr:.3f}  "
              f"(Δ mAP {r['mAP']-fm:+.3f}, Δ rank-1 {r['rank-1']-fr:+.3f})", flush=True)

    hold = args.holdout_camera
    gal_ids = {t.gt_label for t in tracklets if t.camera != hold}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != hold]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gal_ids]
    loco_train = [t.tracklet_id for t in tracklets if t.camera != hold]
    loco_init = {t: ClusterAssigner(0.6, 10).assign(
        loco_train, np.stack([frozen_mean[x] for x in loco_train]),
        {p for p in cl if all(z in loco_train for z in p)})[t] for t in loco_train}

    print("\n========== LEAVE-OUT 66.130: SUPERVISED (ceiling) ==========")
    enc = finetune(loader, loco_train, cl, device, args.steps, args.refresh_every,
                   args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed,
                   mode="supervised", gt=gt)
    show("LEAVE-OUT supervised", query, gallery, enc, "leaveout_supervised")

    print("\n========== LEAVE-OUT 66.130: SSL ==========")
    enc = finetune(loader, loco_train, cl, device, args.steps, args.refresh_every,
                   args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed,
                   mode="ssl", init_labels=loco_init)
    show("LEAVE-OUT ssl", query, gallery, enc, "leaveout_ssl")

    print("\n========== FULL: SSL (transductive) ==========")
    fq, fg = build_full_cross_camera(tracklets)
    full_tids = sorted({it.tracklet_id for it in fq})
    full_init = {t: ClusterAssigner(0.6, 10).assign(
        full_tids, np.stack([frozen_mean[x] for x in full_tids]),
        {p for p in cl if all(z in full_tids for z in p)})[t] for t in full_tids}
    enc = finetune(loader, full_tids, cl, device, args.steps, args.refresh_every,
                   args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed,
                   mode="ssl", init_labels=full_init)
    show("FULL ssl", fq, fg, enc, "full_ssl")


if __name__ == "__main__":
    main()
