"""Frozen MegaDescriptor-DINOv2 (ViT-L/14@518, animal-domain) retrieval — isolates
whether the s70 collapse was the backbone or training divergence. Native 518, no
resize. Also checks the s70 checkpoint embeddings for NaN.

    python megadino_frozen.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader
from vitb_unsup_megadino import megadino_backbone

OBL = "66.130"


@torch.no_grad()
def embed(model, cload, tids, device, flip=False, bs=8):
    out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        rows, counts = [], []
        for t in chunk:
            r = cload._clip_rows(t, 8, train=False)
            rows += r; counts.append(len(r))
        x = cload._norm(rows)
        if flip:
            x = torch.flip(x, dims=[-1])
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(x).float()
        f = F.normalize(f, dim=1)
        off = 0
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

    gal_ids_p1 = {t.gt_label for t in tracklets if t.camera != OBL}
    ids = sorted({t.tracklet_id for t in tracklets
                  if t.camera != OBL or t.gt_label in gal_ids_p1})

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    bb = megadino_backbone(n_blocks=0, load_weights=True).to(device).eval()
    print("frozen animal-DINOv2 ViT-L, embedding...", flush=True)
    E0 = embed(bb, cload, ids, device, flip=False)
    E1 = embed(bb, cload, ids, device, flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Emb = np.stack([nrm(nrm(E0[t]) + nrm(E1[t])) for t in ids])
    print("has NaN:", bool(np.isnan(Emb).any()), flush=True)
    np.savez_compressed("_megadino_frozen_emb.npz", ids=np.array(ids), m0=Emb)
    embs = [Emb]

    def run(q, g, name):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        r = _score(q, g, champ_dist(q, g, embs, ids, cams_qg), (1, 5, 10))
        print(f"  {name:10s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
              f"mAP={r['mAP']:.3f}", flush=True)
        return r

    print("\nFROZEN animal-DINOv2 ViT-L (champion recipe):", flush=True)
    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    run(q1, g1, "P1")
    dm = []
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gset = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gset]
        if q:
            dm.append(run(q, g, f"q_{X}")["rank-1"])
    print(f"  dorsal mean: {np.mean(dm):.3f}", flush=True)
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
    run(items, items, "P2")
    print("\nrefs: frozen Mega-L-384 P1 0.626/dorsal 0.50/P2 0.46; "
          "frozen DINOv2 P1 0.62", flush=True)


if __name__ == "__main__":
    main()
