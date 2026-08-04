"""Frozen SigLIP2 so400m (3rd backbone family) retrieval — does an image-text
contrastive encoder add a complementary view for the ensemble? Native 384, GAP
pool, 1152-d. Zero training. Champion recipe + flip TTA on all protocols.

    python siglip_frozen.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import timm

from cowreid import Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader

OBL = "66.130"
MODEL = "vit_so400m_patch16_siglip_gap_384.v2_webli"
IMG = 384


@torch.no_grad()
def embed(model, cload, tids, device, flip=False, bs=8):
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)   # SigLIP norm
    out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        rows, counts = [], []
        for t in chunk:
            r = cload._clip_rows(t, 8, train=False); rows += r; counts.append(len(r))
        u8 = torch.from_numpy(np.ascontiguousarray(cload.cache[rows])).permute(0, 3, 1, 2)
        x = u8.to(device).float().div_(255.0)
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        if flip:
            x = torch.flip(x, dims=[-1])
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(x).float()
        f = F.normalize(f, dim=1); off = 0
        for t, c in zip(chunk, counts):
            out[t] = f[off:off + c].mean(0).cpu().numpy(); off += c
    return out


def main():
    device = "cuda"
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    gal = {t.gt_label for t in tracklets if t.camera != OBL}
    ids = sorted({t.tracklet_id for t in tracklets if t.camera != OBL or t.gt_label in gal})

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))
    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(device).eval()
    print(f"loaded {MODEL}, embedding...", flush=True)
    E0 = embed(model, cload, ids, device, flip=False)
    E1 = embed(model, cload, ids, device, flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Emb = np.stack([nrm(nrm(E0[t]) + nrm(E1[t])) for t in ids])
    np.savez_compressed("_siglip_frozen_emb.npz", ids=np.array(ids), m0=Emb)
    embs = [Emb]

    def run(q, g, name):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        r = _score(q, g, champ_dist(q, g, embs, ids, cams_qg), (1, 5, 10))
        print(f"  {name:10s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}", flush=True)
        return r

    print("\nFROZEN SigLIP2 so400m (champion recipe):", flush=True)
    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    r_p1 = run(q1, g1, "P1")
    dm = []
    per_cam = {}
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gs = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gs]
        if q:
            r_cam = run(q, g, f"q_{X}")
            dm.append(r_cam["rank-1"])
            per_cam[X] = r_cam
    print(f"  dorsal mean: {np.mean(dm):.3f}", flush=True)
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
    r_p2 = run(items, items, "P2")
    print("\nrefs: frozen Mega-L-384 P1 0.626/dorsal 0.50/P2 0.46; DINOv2 0.62/0.42/0.34",
          flush=True)

    # serialize all computed metrics (no computation change) to a versioned JSON
    out = {
        "script": "siglip_frozen.py",
        "model": MODEL,
        "img_size": IMG,
        "embeddings_file": "_siglip_frozen_emb.npz",
        "P1": r_p1,
        "dorsal_per_camera": per_cam,
        "dorsal_mean_rank1": round(float(np.mean(dm)), 4),
        "P2": r_p2,
    }
    base = os.path.join("artifacts2", "siglip_frozen")
    v = 1
    while os.path.exists(f"{base}_v{v}.json"):
        v += 1
    path = f"{base}_v{v}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
