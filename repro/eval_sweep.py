"""Evaluate checkpoint(s) on P1 (leave-out 66.130), the per-camera sweep, and P2
(full transductive). Embeds each checkpoint with flip-TTA, fuses by champion-
distance mean, reports all protocols. Standalone process.

    python eval_sweep.py --ckpts _vitb_hcl_s13_ckpt.pt --tag hcl_s13
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_cap_ensemble import embed_tta
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, VITB, CacheLoader

OBL = "66.130"


def champ_dist(q, g, embs_list, ids, cams_qg):
    dists = []
    for M in embs_list:
        E = {t: M[i] for i, t in enumerate(ids)}
        Qf = np.stack([E[it.tracklet_id] for it in q])
        Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(list(q) + list(g), X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        dists.append(rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                          dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20))
    return np.mean(dists, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--backbone", choices=["dinov2", "dino3", "mega", "megadino"], default="dinov2")
    args = ap.parse_args()
    device = "cuda"

    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    by_tid = {t.tracklet_id: t for t in tracklets}

    # eval universe = all tracklets embeddable (same as prior npz coverage)
    gal_ids_p1 = {t.gt_label for t in tracklets if t.camera != OBL}
    ids = sorted({t.tracklet_id for t in tracklets
                  if t.camera != OBL or t.gt_label in gal_ids_p1})

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(0))

    embs = []
    for ckpt in args.ckpts:
        ck = torch.load(ckpt, map_location="cpu")
        n_cls = n_cls_from_ckpt(ck["model"])
        if args.backbone == "dino3":
            from vitb_unsup_dino3 import DINO3, Dino3Student
            backbone = DinoV2Backbone(model_name=DINO3, pretrained=False).requires_grad_(False)
            model = Dino3Student(backbone, n_cls, 256).to(device)
        elif args.backbone == "mega":
            from vitb_unsup_mega import MegaBackbone, MegaStudent
            backbone = MegaBackbone(pretrained=False, n_stage=1).requires_grad_(False)
            model = MegaStudent(backbone, n_cls, 256).to(device)
        elif args.backbone == "megadino":
            from vitb_unsup_megadino import megadino_backbone
            backbone = megadino_backbone(n_blocks=4, load_weights=False).requires_grad_(False)
            model = FineTuneIICS(backbone, n_cls, 256).to(device)
        else:
            backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
            model = FineTuneIICS(backbone, n_cls, 256).to(device)
        model.load_state_dict(ck["model"])
        print(f"[{ckpt}] step={ck['step']} embedding (TTA)...", flush=True)
        E = embed_tta(model, cload, ids, args.frames, device)
        embs.append(np.stack([E[t] for t in ids]))
        del model, backbone; torch.cuda.empty_cache()
    np.savez_compressed(f"_sweep_{args.tag}_emb.npz", ids=np.array(ids),
                        **{f"m{i}": M for i, M in enumerate(embs)})

    report = {}
    # P1
    g1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and by_tid[t].gt_label in {x.identity for x in g1}]
    cams_qg = [it.camera for it in q1] + [it.camera for it in g1]
    r = _score(q1, g1, champ_dist(q1, g1, embs, ids, cams_qg), (1, 5, 10))
    print(f"\nP1 leave-out {OBL}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
          f"mAP={r['mAP']:.3f}", flush=True)
    report["P1"] = r

    # per-camera sweep + dorsal mean
    dorsal_r1 = []
    for X in sorted({by_tid[t].camera for t in ids}):
        g = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
             if by_tid[t].camera != X]
        gset = {it.identity for it in g}
        q = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and by_tid[t].gt_label in gset]
        if not q:
            continue
        cams_qg = [it.camera for it in q] + [it.camera for it in g]
        r = _score(q, g, champ_dist(q, g, embs, ids, cams_qg), (1, 5, 10))
        print(f"  query={X:8s} r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
              f"mAP={r['mAP']:.3f}", flush=True)
        report[f"query_{X}"] = r
        if X != OBL:
            dorsal_r1.append(r["rank-1"])
    print(f"  dorsal-query mean r1 = {np.mean(dorsal_r1):.3f}", flush=True)
    report["dorsal_mean_r1"] = float(np.mean(dorsal_r1))

    # P2
    id_cams = defaultdict(set)
    for t in ids:
        id_cams[by_tid[t].gt_label].add(by_tid[t].camera)
    multi = {i for i, c in id_cams.items() if len(c) >= 2}
    items = [EvalItem(t, by_tid[t].gt_label, by_tid[t].camera) for t in ids
             if by_tid[t].gt_label in multi]
    cams_qg = [it.camera for it in items] * 2
    r = _score(items, items, champ_dist(items, items, embs, ids, cams_qg), (1, 5, 10))
    print(f"P2 full transductive: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} "
          f"mAP={r['mAP']:.3f}", flush=True)
    report["P2"] = r

    with open(f"artifacts2/sweep_{args.tag}_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"saved artifacts2/sweep_{args.tag}_v1.json", flush=True)
    print("refs: baseline trio P1 0.883, dorsal mean 0.511, P2 0.516;"
          " deploy trio P2 0.549", flush=True)


if __name__ == "__main__":
    main()
