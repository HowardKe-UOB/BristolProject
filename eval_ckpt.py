"""Standalone eval of any FineTuneIICS ViT-B checkpoint: feat768 read-out, normal
+ horizontal-flip TTA, champion recipe RRF(CC,PCAW,CC-RR). Clean process (no
training state) to avoid the memory/IO hang seen when eval ran inside the trainer.

    python eval_ckpt.py --ckpt _vitb_strict_ckpt.pt --tag strict
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader


@torch.no_grad()
def feat768(model, cload, tids, T, device, flip=False, bs=8):
    model.eval(); out = {}
    for i in range(0, len(tids), bs):
        ch = tids[i:i + bs]; x = cload.batch(ch, T, train=False)
        if flip:
            x = torch.flip(x, dims=[-1])
        with torch.autocast("cuda", dtype=torch.float16):
            e = F.normalize(model.head.pool(model._frames(x)), dim=1)
        for t, v in zip(ch, e.float().cpu().numpy()):
            out[t] = v
        del x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="ckpt")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"

    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
    q, g = list(query), list(gallery)
    cams_qg = [it.camera for it in q] + [it.camera for it in g]
    eval_tids = sorted({it.tracklet_id for it in q + g})
    mask = build_st_mask(q, g, index, topo, margin=0)

    ck = torch.load(args.ckpt, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(0))

    print(f"{args.tag}: ckpt={args.ckpt} step={ck['step']}  embedding...", flush=True)
    E0 = feat768(model, cload, eval_tids, args.frames, device, flip=False)
    E1 = feat768(model, cload, eval_tids, args.frames, device, flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Etta = {t: nrm(nrm(E0[t]) + nrm(E1[t])) for t in eval_tids}

    def champ(E):
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    report = {"ckpt": args.ckpt, "step": int(ck["step"])}
    print(f">>> {args.tag} (step {ck['step']}), leave-out {HOLD}", flush=True)
    for name, E in [("champion recipe", E0), ("+ flip TTA", Etta)]:
        dist = champ(E)
        r = _score(q, g, dist, (1, 5, 10)); dm = dist.copy(); dm[mask] = INF
        rs = _score(q, g, dm, (1, 5, 10))
        print(f"    {name:18s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
              f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
        report[name] = {"plain": r, "st": rs}
    outp = f"artifacts2/strict_eval_{args.tag}_v1.json"
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {outp}   (champion ref: no-TTA 0.706 / +TTA 0.718)", flush=True)


if __name__ == "__main__":
    main()
