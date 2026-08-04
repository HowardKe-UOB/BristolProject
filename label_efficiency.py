"""Label-efficiency curve (semi-supervised #3) on leave-out 66.130.

Enroll a fraction f of the training identities (label ALL their tracklets with GT;
the rest stay unlabeled and are clustered). Train the same fine-tuned backbone +
IICS pipeline semi-supervised, eval cross-view, multi-seed. f=0 -> unsupervised,
f=1 -> fully supervised (~0.847). Shows how little labeling crosses rank-1 0.80.

    python label_efficiency.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz \
        --fractions 0 0.1 0.25 0.5 1.0 --seeds 0 1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets, extract_paths
from cowreid.cluster import build_cannot_link
from cowreid.eval import EvalItem, ReIDEvaluator
from cowreid.tracklets import TrackletIndex
from train_finetune import ClipLoader
from train_finetune_iics import embed_all, train
from train_phase2_run import sample_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--refresh-every", type=int, default=200)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
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
    loader = ClipLoader(index, args.work, args.frames, 518, device, np.random.default_rng(0))
    d = np.load(args.cache, allow_pickle=True)
    fc = {k: v for k, v in zip(d["ids"], d["clips"])}
    frozen_mean = {t: (fc[t].mean(0) / (np.linalg.norm(fc[t].mean(0)) + 1e-12)) for t in fc}

    hold = args.holdout_camera
    gal_ids = {t.gt_label for t in tracklets if t.camera != hold}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != hold]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == hold and t.gt_label in gal_ids]
    loco_train = [t.tracklet_id for t in tracklets if t.camera != hold]
    eval_tids = sorted({it.tracklet_id for it in query + gallery})
    train_ids = sorted({gt[t] for t in loco_train})
    ev = ReIDEvaluator(ranks=(1, 5, 10))

    curve = {}
    for f in args.fractions:
        runs = []
        for seed in args.seeds:
            rng = np.random.default_rng(1000 * seed + int(round(f * 100)))
            ids = train_ids.copy(); rng.shuffle(ids)
            enrolled = set(ids[: int(round(f * len(ids)))])
            labeled = {t for t in loco_train if gt[t] in enrolled}
            torch.manual_seed(seed); np.random.seed(seed)
            loader.rng = np.random.default_rng(seed)
            model = train(loader, loco_train, index, cl, device, args.steps, args.refresh_every,
                          args.P, args.K, args.T, args.proj_dim, args.n_blocks, seed, frozen_mean,
                          eps=args.eps, topo=topo, gt=gt, crossview=False, labeled_tids=labeled)
            r = ev.evaluate(query, gallery, embed_all(model, loader, eval_tids, args.frames, device))
            runs.append(r)
            print(f"[f={f:.2f} seed={seed}] enrolled={len(enrolled)}/{len(ids)} ids "
                  f"({len(labeled)} tracklets)  rank-1={r['rank-1']:.3f} rank-5={r['rank-5']:.3f} "
                  f"mAP={r['mAP']:.3f}", flush=True)
        curve[f] = runs

    print("\n===== LABEL-EFFICIENCY CURVE (leave-out 66.130, semi-supervised) =====")
    print(f"{'frac':>6} | {'rank-1':>14} | {'rank-5':>14} | {'mAP':>14}")
    for f in args.fractions:
        runs = curve[f]
        def ms(key):
            v = np.array([r[key] for r in runs])
            return f"{v.mean():.3f}+/-{v.std():.3f}"
        print(f"{f:>6.2f} | {ms('rank-1'):>14} | {ms('rank-5'):>14} | {ms('mAP'):>14}")
    print("(f=0 unsupervised ref ~0.47 rank-1; f=1 supervised ceiling 0.847/0.951)")


if __name__ == "__main__":
    main()
