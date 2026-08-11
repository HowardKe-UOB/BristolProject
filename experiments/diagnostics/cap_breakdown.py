"""Confirm the CAP breakthrough is robust across ALL inference levers (not a
single-recipe fluke). Embed the CAP checkpoint's feat768 (normal + flip-TTA),
save to npz, and run the full lever matrix (cosine/CC/PCAW/CC-RR/RRF, each ± ST)
via st_validate2.run_all, side by side with the prior champion.

    python cap_breakdown.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from st_eval_vitb import n_cls_from_ckpt
from st_validate2 import run_all
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader

CAP_CKPT = "_vitb_cap_ckpt.pt"
EMB_NPZ = "_vitb_cap_emb_v1.npz"


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
    eval_tids = sorted({it.tracklet_id for it in q + g})

    ck = torch.load(CAP_CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    print(f"CAP ckpt step={ck['step']}, embedding feat768 (normal + flip)...", flush=True)
    E0 = feat768(model, cload, eval_tids, 8, device, flip=False)
    E1 = feat768(model, cload, eval_tids, 8, device, flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Etta = {t: nrm(nrm(E0[t]) + nrm(E1[t])) for t in eval_tids}
    np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                        feat768=np.stack([E0[t] for t in eval_tids]),
                        feat768_tta=np.stack([Etta[t] for t in eval_tids]))
    print(f"saved {EMB_NPZ}\n", flush=True)

    report = {}
    print("=== CAP feat768 (no TTA) — full lever matrix ===", flush=True)
    run_all(q, g, E0, index, topo, margin=0, tag="CAP/", report=report)
    print("\n=== CAP feat768 + flip TTA — full lever matrix ===", flush=True)
    run_all(q, g, Etta, index, topo, margin=0, tag="CAP+TTA/", report=report)

    with open("artifacts2/cap_breakdown_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nprior champion+TTA ref: r1=0.718 r5=0.877 mAP=0.425")
    print("saved artifacts2/cap_breakdown_v1.json")


if __name__ == "__main__":
    main()
