"""Apply the SOTA levers harder: bigger backbone (DINOv2 ViT-B) + k-reciprocal
(camera-aware Jaccard) re-ranking at eval + longer training. Run SUPERVISED (all
labels) and UNSUPERVISED (CA-Jaccard + crop-OT mining) on leave-out 66.130, and
report BOTH cosine and re-ranked metrics. Tests whether the fuller config pushes
supervised toward 0.90 and lifts unsupervised.

    python push_sota.py --listing 2025Sep18.listing.txt --tar 2025Sep18.tar.gz
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import argparse

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets, extract_paths
from cowreid.cluster import build_cannot_link
from cowreid.crossview_ot import crossview_crop_bags
from cowreid.eval import EvalItem, ReIDEvaluator, evaluate_rerank
from cowreid.tracklets import TrackletIndex
from train_finetune import ClipLoader
from train_finetune_iics import embed_all, train
from train_phase2_run import sample_frames

VITB = "vit_base_patch14_dinov2.lvd142m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", required=True)
    ap.add_argument("--tar", required=True)
    ap.add_argument("--work", default="_crops_train")
    ap.add_argument("--cache", default="dino_clip_feats_v1.npz")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--refresh-every", type=int, default=300)
    ap.add_argument("--P", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--backbone", default=VITB)
    ap.add_argument("--max-bags", type=int, default=2000)
    ap.add_argument("--holdout-camera", default="66.130")
    ap.add_argument("--mode", choices=["both", "supervised", "unsupervised"], default="both")
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
    loader = ClipLoader(index, args.work, args.frames, 518, device, np.random.default_rng(args.seed))
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

    train_cams = [c for c in manifest.cameras() if c != hold]
    crop_bags, crop_paths = crossview_crop_bags(manifest, topo, train_cams, index,
                                                max_bags=args.max_bags)
    extract_paths(args.tar, crop_paths, args.work)

    def run(tag, **kw):
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        loader.rng = np.random.default_rng(args.seed)
        model = train(loader, loco_train, index, cl, device, args.steps, args.refresh_every,
                      args.P, args.K, args.T, args.proj_dim, args.n_blocks, args.seed, frozen_mean,
                      eps=args.eps, topo=topo, gt=gt, model_name=args.backbone, **kw)
        emb = embed_all(model, loader, eval_tids, args.frames, device)
        raw = ev.evaluate(query, gallery, emb)
        rr = evaluate_rerank(query, gallery, emb)
        print(f"\n>>> {tag} ({args.backbone})", flush=True)
        print(f"    cosine : mAP={raw['mAP']:.3f} rank-1={raw['rank-1']:.3f} rank-5={raw['rank-5']:.3f}", flush=True)
        print(f"    rerank : mAP={rr['mAP']:.3f} rank-1={rr['rank-1']:.3f} rank-5={rr['rank-5']:.3f}", flush=True)

    if args.mode in ("both", "supervised"):
        print("========== SUPERVISED (ViT-B, all labels) ==========")
        run("SUPERVISED", crossview=False, labeled_tids=set(loco_train))

    if args.mode in ("both", "unsupervised"):
        print("\n========== UNSUPERVISED (ViT-B, CA-Jaccard + crop-OT) ==========")
        run("UNSUPERVISED", crossview=True, mine_mode="crop_ot", crop_bags=crop_bags,
            crop_paths=crop_paths)


if __name__ == "__main__":
    main()
