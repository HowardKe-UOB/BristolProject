"""Multi-seed confirmation of the best label-free config on leave-out 66.130.

Runs the same pipeline (fine-tuned backbone + IICS + CA-Jaccard + cross-view
positive re-mining) across several seeds and reports mean +/- std for mAP /
rank-1 / rank-5 -- the error bars a paper needs. Setup (manifest, tracklets,
crops, frozen-feature init) is built once; only the seed varies per run.

    python multiseed_leaveout.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz --seeds 0 1 2 3
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets, extract_paths
from cowreid.cluster import build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags
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
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--crossview-sim", type=float, default=0.85)
    ap.add_argument("--mine", choices=["tracklet", "crop_ot"], default="crop_ot")
    ap.add_argument("--max-bags", type=int, default=1500)
    ap.add_argument("--ot-min-conf", type=float, default=0.5)
    ap.add_argument("--ot-min-votes", type=int, default=3)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
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
    ev = ReIDEvaluator(ranks=(1, 5, 10))

    # crop-level OT mining setup (extract co-occurring crops on overlapping train pairs)
    crop_bags = crop_paths = None
    if args.mine == "crop_ot":
        train_cams = [c for c in manifest.cameras() if c != hold]
        crop_bags, crop_paths = crossview_crop_bags(manifest, topo, train_cams, index,
                                                    max_bags=args.max_bags)
        print(f"crop-OT: {len(crop_bags)} bags, {len(crop_paths)} crops -> extracting")
        extract_paths(args.tar, crop_paths, args.work)

    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        loader.rng = np.random.default_rng(seed)          # vary train frame sampling
        model = train(loader, loco_train, index, cl, device, args.steps, args.refresh_every,
                      args.P, args.K, args.T, args.proj_dim, args.n_blocks, seed, frozen_mean,
                      eps=args.eps, topo=topo, gt=gt, crossview=True, sim_thr=args.crossview_sim,
                      mine_mode=args.mine, crop_bags=crop_bags, crop_paths=crop_paths,
                      ot_min_conf=args.ot_min_conf, ot_min_votes=args.ot_min_votes)
        r = ev.evaluate(query, gallery, embed_all(model, loader, eval_tids, args.frames, device))
        runs.append(r)
        print(f"\n[seed {seed}] mAP={r['mAP']:.3f} rank-1={r['rank-1']:.3f} "
              f"rank-5={r['rank-5']:.3f}", flush=True)

    print("\n========== MULTI-SEED SUMMARY (leave-out 66.130, label-free) ==========")
    for key in ("mAP", "rank-1", "rank-5", "rank-10"):
        vals = np.array([r[key] for r in runs])
        print(f"  {key:7s} = {vals.mean():.3f} +/- {vals.std():.3f}   "
              f"(runs: {', '.join(f'{v:.3f}' for v in vals)})")
    print(f"  supervised ceiling ref: mAP=0.863 rank-1=0.847 rank-5=0.951")


if __name__ == "__main__":
    main()
