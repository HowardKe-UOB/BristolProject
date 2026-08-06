"""Inference-lever evaluation on the CONVERGED ViT-B unsupervised checkpoint.

No training: rebuilds the model from ``_vitb_unsup_ckpt.pt`` (class counts are
inferred from the checkpoint's per-camera classifier shapes), embeds all eval
tracklets through the uint8 image cache, saves the embeddings to a NEW npz, then
scores every label-free inference combination (camera-centering / ST mask / AQE /
CA-Jaccard re-rank) on both the 256-d projected embedding and the 768-d pooled
backbone feature.

    python st_eval_vitb.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem
from cowreid.tracklets import TrackletIndex
from st_validate2 import run_all
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, CKPT, HOLD, VITB, CacheLoader

EMB_NPZ = "_vitb_unsup_emb_v1.npz"


def n_cls_from_ckpt(state: dict) -> dict[str, int]:
    """Recover {camera: n_classes} from head.classifiers.<cam_key>.weight shapes."""
    out = {}
    for k, v in state.items():
        if k.startswith("head.classifiers.") and k.endswith(".weight"):
            cam = k.split(".")[2].replace("_", ".")
            out[cam] = v.shape[0]
    return out


@torch.no_grad()
def embed_both(model, cload, tids, T, device, bs=16):
    """{tid -> (emb256, feat768)} pooled backbone feature + projected embedding."""
    model.eval()
    E, Ft = {}, {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        x = cload.batch(chunk, T, train=False)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model._frames(x)                    # (B, T, 768)
            pooled = model.head.pool(f)             # (B, 768)
            emb = F.normalize(model.head.embed(pooled), dim=1)
        for k, t in enumerate(chunk):
            E[t] = emb[k].float().cpu().numpy()
            Ft[t] = F.normalize(pooled[k].float(), dim=0).cpu().numpy()
        if (i // bs) % 10 == 0:
            print(f"  embedded {i + len(chunk)}/{len(tids)}", flush=True)
    return E, Ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listing", default="2025Sep18.listing.txt")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default="artifacts2/st_inference_vitb_v1.json")
    args = ap.parse_args()
    device = "cuda"

    manifest = Manifest.from_listing_file(args.listing)
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)

    gal_ids = {t.gt_label for t in tracklets if t.camera != HOLD}
    gallery = [EvalItem(t.tracklet_id, t.gt_label, t.camera) for t in tracklets if t.camera != HOLD]
    query = [EvalItem(t.tracklet_id, t.gt_label, t.camera)
             for t in tracklets if t.camera == HOLD and t.gt_label in gal_ids]
    eval_tids = sorted({it.tracklet_id for it in query + gallery})

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    print(f"checkpoint step={ck['step']}  cameras={sorted(n_cls)}", flush=True)
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r")
    p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(0))

    E, Ft = embed_both(model, cload, eval_tids, args.frames, device)
    np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                        emb256=np.stack([E[t] for t in eval_tids]),
                        feat768=np.stack([Ft[t] for t in eval_tids]))
    print(f"saved {EMB_NPZ}", flush=True)

    report = {"checkpoint_step": int(ck["step"])}
    print(f"\n=== ViT-B unsup, 256-d projected embedding (leave-out {HOLD}) ===", flush=True)
    run_all(query, gallery, E, index, topo, margin=0, tag="emb256/", report=report)
    print(f"\n=== ViT-B unsup, 768-d pooled backbone feature ===", flush=True)
    run_all(query, gallery, Ft, index, topo, margin=0, tag="feat768/", report=report)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
