"""Lever: Test-Time Augmentation (TTA) by horizontal flip, on the ViT-B
unsupervised champion checkpoint. Embed each eval tracklet normally AND
horizontally flipped, average the two L2-normalised embeddings, then apply the
champion recipe RRF(CC, PCAW, CC-RR). Label-free, short GPU job (eval set only).

    python vitb_tta.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, CKPT, HOLD, VITB, CacheLoader

EMB_NPZ = "_vitb_unsup_tta_emb_v1.npz"
RANKS = (1, 5, 10)


@torch.no_grad()
def embed_tta(model, cload, tids, T, device, bs=12, flip=False):
    """768-d pooled backbone feature (the champion read-out), optionally on the
    horizontally-flipped clip."""
    import torch.nn.functional as F
    model.eval(); out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        x = cload.batch(chunk, T, train=False)
        if flip:
            x = torch.flip(x, dims=[-1])                 # horizontal flip (width)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model._frames(x)                         # (B, T, 768)
            pooled = model.head.pool(f)                  # (B, 768) attention pool
        for t, v in zip(chunk, F.normalize(pooled.float(), dim=1).cpu().numpy()):
            out[t] = v
    return out


def champion(q, g, Qf, Gf, cams):
    X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
    Qcc, Gcc = cc[:len(q)], cc[len(q):]
    Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
    return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                dist_rerank(Qcc, Gcc, cams, k1=30, k2=6)], k=20)


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:24s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}


def main():
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
    cams = [it.camera for it in q] + [it.camera for it in g]
    eval_tids = sorted({it.tracklet_id for it in q + g})
    mask = build_st_mask(q, g, index, topo, margin=0)

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r")
    p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    print("embedding (normal)...", flush=True)
    E0 = embed_tta(model, cload, eval_tids, 8, device, flip=False)
    print("embedding (flipped)...", flush=True)
    E1 = embed_tta(model, cload, eval_tids, 8, device, flip=True)

    def norm(v):
        return v / (np.linalg.norm(v) + 1e-12)
    Etta = {t: norm(norm(E0[t]) + norm(E1[t])) for t in eval_tids}
    np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                        feat768=np.stack([Etta[t] for t in eval_tids]))

    def feats(E):
        return (np.stack([E[it.tracklet_id] for it in q]),
                np.stack([E[it.tracklet_id] for it in g]))

    report = {}
    print(f"\nleave-out {HOLD}, ViT-B unsup, champion recipe RRF(CC,PCAW,CC-RR):", flush=True)
    QB, GB = feats(E0); show("no TTA (champion)", q, g, champion(q, g, QB, GB, cams), mask, report)
    QT, GT = feats(Etta); show("flip TTA (champion)", q, g, champion(q, g, QT, GT, cams), mask, report)

    with open("artifacts2/vitb_tta_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nchampion ref (no TTA): r1=0.706 r5=0.859 mAP=0.423")
    print("saved artifacts2/vitb_tta_v1.json")


if __name__ == "__main__":
    main()
