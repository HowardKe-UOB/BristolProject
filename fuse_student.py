"""Fuse the distilled student(s) into the 5-seed CAP ensemble.

Embeds only the new student checkpoint(s) with flip-TTA (short GPU job), saves
them to a new npz, then evaluates distance-mean ensembles of: 5 CAP seeds (ref),
5+student(s), and the ensemble-size trend. CPU for all fusion.

    python fuse_student.py --students _vitb_dst_s5_ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from eval_cap_ensemble import embed_tta
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader

RANKS = (1, 5, 10)


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:32s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="+", default=["_vitb_dst_s5_ckpt.pt"])
    ap.add_argument("--out-npz", default="_vitb_dst_emb_v1.npz")
    ap.add_argument("--out-json", default="artifacts2/fuse_student_v1.json")
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

    # student embeddings (embed only if not cached in out-npz)
    student_emb = {}
    if os.path.exists(args.out_npz):
        dd = np.load(args.out_npz, allow_pickle=True)
        sids = list(dd["ids"])
        for k in dd.files:
            if k != "ids":
                student_emb[k] = np.stack([dd[k][sids.index(t)] for t in eval_tids]) \
                    if sids != eval_tids else dd[k]
    todo = [c for c in args.students
            if os.path.splitext(os.path.basename(c))[0] not in student_emb]
    if todo:
        cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
        cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))
        for ckpt in todo:
            tag = os.path.splitext(os.path.basename(ckpt))[0]
            ck = torch.load(ckpt, map_location="cpu")
            n_cls = n_cls_from_ckpt(ck["model"])
            backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
            model = FineTuneIICS(backbone, n_cls, 256).to(device)
            model.load_state_dict(ck["model"])
            print(f"[{tag}] embedding (TTA)...", flush=True)
            E = embed_tta(model, cload, eval_tids, 8, device)
            student_emb[tag] = np.stack([E[t] for t in eval_tids])
            del model, backbone; torch.cuda.empty_cache()
        np.savez_compressed(args.out_npz, ids=np.array(eval_tids), **student_emb)
        print(f"saved {args.out_npz}", flush=True)

    d = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    ids = list(d["ids"]); pos = {t: i for i, t in enumerate(ids)}
    order = [pos[t] for t in eval_tids]
    cap = {s: d[s][order] for s in d.files if s != "ids"}

    def champ_dist(M):
        E = {t: M[i] for i, t in enumerate(eval_tids)}
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    print("computing champion distances...", flush=True)
    dcap = {s: champ_dist(M) for s, M in cap.items()}
    dstu = {s: champ_dist(M) for s, M in student_emb.items()}

    report = {}
    print(f"\nleave-out {HOLD}:", flush=True)
    for s, dm in dstu.items():
        show(f"student {s}", q, g, dm, mask, report)
    show("ens CAP5 (ref)", q, g, np.mean(list(dcap.values()), axis=0), mask, report)
    show("ens CAP5+students", q, g,
         np.mean(list(dcap.values()) + list(dstu.values()), axis=0), mask, report)
    # student-weighted (students trained on cleaner labels -> try double weight)
    show("ens CAP5+students x2", q, g,
         np.mean(list(dcap.values()) + list(dstu.values()) * 2, axis=0), mask, report)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nrefs: CAP5 ens 0.749/0.932/0.491 | goal 0.80")
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
