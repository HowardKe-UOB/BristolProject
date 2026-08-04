"""Evaluate the DEPLOYMENT-MODE trio (trained on all 7 cameras) on the full
transductive protocol and the per-camera query sweep. Embeds the 3 checkpoints
with flip-TTA (GPU), saves a new npz, then CPU-fuses exactly like
validate_protocols.py did for the holdout-mode trio.

    python validate_deploy.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from eval_cap_ensemble import embed_tta
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from validate_protocols import champ_dist
from vitb_unsup import CACHE_JSON, CACHE_NPY, VITB, CacheLoader

CKPTS = ["_vitb_dep_s10_ckpt.pt", "_vitb_dep_s11_ckpt.pt", "_vitb_dep_s12_ckpt.pt"]
EMB_NPZ = "_vitb_dep_emb_v1.npz"


def main():
    device = "cuda"
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    topo = CameraTopology.from_gt(manifest)
    by_tid = {t.tracklet_id: t for t in tracklets}

    ref = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(ref["ids"])                              # same 997 tracklets

    import os
    if os.path.exists(EMB_NPZ):
        d = np.load(EMB_NPZ, allow_pickle=True)
        embs = [d[k] for k in sorted(d.files) if k != "ids"]
        print(f"loaded {EMB_NPZ}", flush=True)
    else:
        cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
        cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))
        out = {}
        for ckpt in CKPTS:
            tag = ckpt.replace(".pt", "").strip("_")
            ck = torch.load(ckpt, map_location="cpu")
            n_cls = n_cls_from_ckpt(ck["model"])
            backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
            model = FineTuneIICS(backbone, n_cls, 256).to(device)
            model.load_state_dict(ck["model"])
            print(f"[{tag}] embedding (TTA)...", flush=True)
            E = embed_tta(model, cload, ids, 8, device)
            out[tag] = np.stack([E[t] for t in ids])
            del model, backbone; torch.cuda.empty_cache()
        np.savez_compressed(EMB_NPZ, ids=np.array(ids), **out)
        print(f"saved {EMB_NPZ}", flush=True)
        embs = [out[k] for k in sorted(out)]

    id_cams = defaultdict(set)
    for t in ids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}

    report = {}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
             for t in ids if by_tid[t].gt_label in multi]
    cams_qg = [it.camera for it in items] * 2
    print(f"\n[A] FULL transductive ({len(items)} items), DEPLOY trio:", flush=True)
    dist = champ_dist(items, items, embs, ids, index, topo, cams_qg)
    mask = build_st_mask(items, items, index, topo, margin=0)
    r = _score(items, items, dist, (1, 5, 10))
    dm = dist.copy(); dm[mask] = INF
    rs = _score(items, items, dm, (1, 5, 10))
    print(f"    r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} r10={r['rank-10']:.3f} "
          f"mAP={r['mAP']:.3f}  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} "
          f"mAP={rs['mAP']:.3f}", flush=True)
    report["full_transductive"] = {"plain": r, "st": rs}

    print(f"\n[B] per-camera query sweep, DEPLOY trio:", flush=True)
    for X in sorted({by_tid[t].camera for t in ids}):
        g_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera != X]
        g_set = {it.identity for it in g_items}
        q_items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera)
                   for t in ids if by_tid[t].camera == X and by_tid[t].gt_label in g_set]
        if not q_items:
            continue
        cams_qg = [it.camera for it in q_items] + [it.camera for it in g_items]
        dist = champ_dist(q_items, g_items, embs, ids, index, topo, cams_qg)
        mask = build_st_mask(q_items, g_items, index, topo, margin=0)
        r = _score(q_items, g_items, dist, (1, 5, 10))
        dm = dist.copy(); dm[mask] = INF
        rs = _score(q_items, g_items, dm, (1, 5, 10))
        print(f"    query={X:7s} |Q|={len(q_items):4d}: r1={r['rank-1']:.3f} "
              f"r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}  |+ST r1={rs['rank-1']:.3f}",
              flush=True)
        report[f"query_{X}"] = {"plain": r, "st": rs, "n_query": len(q_items)}

    with open("artifacts2/validate_deploy_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nrefs (holdout-mode trio): full-trans 0.516; per-cam 0.42-0.56 (66.130: 0.877)")
    print("saved artifacts2/validate_deploy_v1.json")


if __name__ == "__main__":
    main()
