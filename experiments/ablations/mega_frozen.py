"""FROZEN MegaDescriptor-L-384 features: does an animal-ReID foundation model beat
frozen DINOv2 on our cattle data? Zero training. Embeds eval tracklets (mean over
8 frames, flip-TTA) at 384, then scores P1 / dorsal sweep / P2 with the champion
inference recipe. Decision gate for a full retrain.

    python mega_frozen.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import timm

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist
from train_phase2_run import sample_frames
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader

OBL = "66.130"
MODEL = "hf-hub:BVRA/MegaDescriptor-L-384"
IMG = 384


@torch.no_grad()
def embed(model, cload, tids, device, flip=False, bs=8):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        rows = []
        counts = []
        for t in chunk:
            r = cload._clip_rows(t, 8, train=False)
            rows += r; counts.append(len(r))
        u8 = torch.from_numpy(np.ascontiguousarray(cload.cache[rows])).permute(0, 3, 1, 2)
        x = u8.to(device).float().div_(255.0)
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
        x = (x - mean) / std
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

    model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(device).eval()
    print("embedding (normal)...", flush=True)
    E0 = embed(model, cload, ids, device, flip=False)
    print("embedding (flip)...", flush=True)
    E1 = embed(model, cload, ids, device, flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Emb = np.stack([nrm(nrm(E0[t]) + nrm(E1[t])) for t in ids])
    np.savez_compressed("_mega_frozen_emb.npz", ids=np.array(ids), m0=Emb)
    embs = [Emb]

    report = {}

    def run(q, g, name):
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        r = _score(q, g, champ_dist(q, g, embs, ids, cams_qg), (1, 5, 10))
        print(f"  {name:10s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
              f"mAP={r['mAP']:.3f}", flush=True)
        report[name] = r
        return r

    print("\nMegaDescriptor-L-384 FROZEN (champion recipe), vs DINOv2-frozen refs:",
          flush=True)
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
    report["dorsal_mean"] = float(np.mean(dm))
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[gt[t]].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if gt[t] in multi]
    run(items, items, "P2")

    with open("artifacts2/mega_frozen_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nDINOv2-frozen refs (champion recipe): P1~0.62 dorsal~0.42 P2~0.34;"
          " trained DINOv2 P1 0.883/P2 0.585", flush=True)
    print("saved artifacts2/mega_frozen_v1.json", flush=True)


if __name__ == "__main__":
    main()
