"""DIAGNOSTIC (no training): does STRICTER cross-camera crop mining actually raise
must-link PRECISION on the champion ViT-B checkpoint? The two failed attacks both
raised recall at fixed ~15% precision. Here we sweep strictness (dustbin
confidence, vote count, and an added cosine gate) and report (#links, precision)
in BOTH crop-embedding spaces (emb256 projection, feat768 backbone). GT is used to
MEASURE precision only. If some setting yields enough links at much higher
precision, a retrain is worthwhile; else strict mining can't help either.

    python strict_mine_diag.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.crossview_ot import crossview_crop_bags
from cowreid.encoder import DinoV2Backbone
from cowreid.sinkhorn import match_with_dustbin
from cowreid.tracklets import TrackletIndex
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, CKPT, HOLD, VITB, CacheLoader


def mine_strict(bags, crop_emb, path_to_tracklet, min_conf, min_votes, cos_thr, gt):
    """crop-OT must-links with an added cosine gate (accept a matched pair only if
    the crops' cosine similarity >= cos_thr). Returns (links, precision, n_cand)."""
    votes = defaultdict(float); counts = defaultdict(int)
    for _cA, _cB, A, B in bags:
        EA = np.stack([crop_emb[p] for p in A]); EB = np.stack([crop_emb[p] for p in B])
        EA = EA / (np.linalg.norm(EA, axis=1, keepdims=True) + 1e-12)
        EB = EB / (np.linalg.norm(EB, axis=1, keepdims=True) + 1e-12)
        sim = EA @ EB.T
        cost = 1.0 - sim
        for i, j, conf in match_with_dustbin(cost, eps=0.1):
            if conf < min_conf or sim[i, j] < cos_thr:
                continue
            ta, tb = path_to_tracklet(A[i]), path_to_tracklet(B[j])
            if ta and tb and ta != tb:
                k = frozenset((ta, tb)); votes[k] += conf; counts[k] += 1
    links = [k for k in votes if counts[k] >= min_votes]
    prec = float(np.mean([gt[tuple(k)[0]] == gt[tuple(k)[1]] for k in links])) if links else None
    n_true = sum(gt[tuple(k)[0]] == gt[tuple(k)[1]] for k in links)
    return links, prec, n_true


@torch.no_grad()
def embed_crops_space(model, cload, paths, device, space, bs=48):
    model.eval(); out = {}
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        x = cload.crops(chunk)                          # (N, 1, 3, H, W)
        with torch.autocast("cuda", dtype=torch.float16):
            if space == "emb256":
                e = model.embed(x)
            else:                                       # feat768 pooled backbone
                e = F.normalize(model.head.pool(model._frames(x)), dim=1)
        for p, v in zip(chunk, e.float().cpu().numpy()):
            out[p] = v
    return out


def main():
    device = "cuda"
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    train_cams = [c for c in {t.camera for t in tracklets} if c != HOLD]
    bags, crop_paths = crossview_crop_bags(
        Manifest([s for t in tracklets for s in t.samples]), topo, train_cams, index,
        max_bags=2000)
    print(f"{len(bags)} bags, {len(crop_paths)} crops", flush=True)

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = __import__("json").load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    print("champion mining ref: min_conf=0.5 min_votes=3 cos_thr=-1 (loose)\n", flush=True)
    for space in ("emb256", "feat768"):
        ce = embed_crops_space(model, cload, crop_paths, device, space)
        print(f"===== crop space = {space} =====", flush=True)
        print(f"  {'setting':38s}  #links  prec   #true", flush=True)
        configs = [
            ("loose (champion)", 0.5, 3, -1.0),
            ("conf0.6 v4", 0.6, 4, -1.0),
            ("conf0.7 v5", 0.7, 5, -1.0),
            ("conf0.5 v3 cos0.5", 0.5, 3, 0.5),
            ("conf0.6 v4 cos0.6", 0.6, 4, 0.6),
            ("conf0.7 v5 cos0.7", 0.7, 5, 0.7),
            ("conf0.7 v6 cos0.75", 0.7, 6, 0.75),
            ("conf0.8 v5 cos0.8", 0.8, 5, 0.8),
        ]
        for name, mc, mv, ct in configs:
            links, prec, ntrue = mine_strict(bags, ce, index.tracklet_of, mc, mv, ct, gt)
            ps = f"{prec:.3f}" if prec is not None else "  -  "
            print(f"  {name:38s}  {len(links):5d}   {ps}   {ntrue}", flush=True)
        print()


if __name__ == "__main__":
    main()
