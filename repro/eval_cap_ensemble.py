"""3-seed CAP ENSEMBLE at inference (label-free variance reduction).

Embeds the eval tracklets with each of the three CAP seed checkpoints
(normal + horizontal-flip TTA averaged), then fuses:
  (a) feature concat  -> champion recipe RRF(CC, PCAW, CC-RR)
  (b) RRF across the three per-seed champion distance matrices
  (c) mean of the three per-seed champion distance matrices
Saves per-seed embeddings to ONE new npz so future fusions are CPU-only.

    python eval_cap_ensemble.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf
from st_eval_vitb import n_cls_from_ckpt
from train_finetune_iics import FineTuneIICS
from vitb_unsup import CACHE_JSON, CACHE_NPY, HOLD, VITB, CacheLoader

SEEDS = {"s0": "_vitb_cap_s0_ckpt.pt", "s1": "_vitb_cap_s1_ckpt.pt", "s2": "_vitb_cap_s2_ckpt.pt",
         "s3": "_vitb_cap_s3_ckpt.pt", "s4": "_vitb_cap_s4_ckpt.pt"}
EMB_NPZ = "_vitb_cap_ens5_emb_v1.npz"
RANKS = (1, 5, 10)


@torch.no_grad()
def embed_tta(model, cload, tids, T, device, bs=8):
    model.eval(); out = {}
    for flip in (False, True):
        for i in range(0, len(tids), bs):
            ch = tids[i:i + bs]; x = cload.batch(ch, T, train=False)
            if flip:
                x = torch.flip(x, dims=[-1])
            with torch.autocast("cuda", dtype=torch.float16):
                e = F.normalize(model.head.pool(model._frames(x)), dim=1)
            for t, v in zip(ch, e.float().cpu().numpy()):
                out[t] = out.get(t, 0) + v / (np.linalg.norm(v) + 1e-12)
            del x
    return {t: v / (np.linalg.norm(v) + 1e-12) for t, v in out.items()}


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:30s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
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
    cams_qg = [it.camera for it in q] + [it.camera for it in g]
    eval_tids = sorted({it.tracklet_id for it in q + g})
    mask = build_st_mask(q, g, index, topo, margin=0)

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    embs = {}
    for tag, ckpt in SEEDS.items():
        ck = torch.load(ckpt, map_location="cpu")
        n_cls = n_cls_from_ckpt(ck["model"])
        backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
        model = FineTuneIICS(backbone, n_cls, 256).to(device)
        model.load_state_dict(ck["model"])
        print(f"[{tag}] step={ck['step']} embedding (TTA)...", flush=True)
        embs[tag] = embed_tta(model, cload, eval_tids, 8, device)
        del model, backbone
        torch.cuda.empty_cache()
    np.savez_compressed(EMB_NPZ, ids=np.array(eval_tids),
                        **{tag: np.stack([E[t] for t in eval_tids]) for tag, E in embs.items()})
    print(f"saved {EMB_NPZ}", flush=True)

    def champ_dist(E):
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    report = {}
    print(f"\nleave-out {HOLD}: per-seed (champion recipe + TTA)", flush=True)
    dists = {}
    for tag in SEEDS:
        dists[tag] = champ_dist(embs[tag])
        show(f"CAP {tag}", q, g, dists[tag], mask, report)

    print("\n-- 3-seed ensembles --", flush=True)
    # (a) feature concat (equal weight; each already L2 per seed)
    Econcat = {t: np.concatenate([embs[tag][t] for tag in SEEDS]) for t in eval_tids}
    show("ens concat->champion", q, g, champ_dist(Econcat), mask, report)
    # (b) RRF across per-seed champion distances
    show("ens RRF(3 champion dists)", q, g, rrf(list(dists.values()), k=20), mask, report)
    # (c) mean of per-seed champion distances (they are RRF scores, same scale)
    show("ens mean(3 champion dists)", q, g, np.mean(list(dists.values()), axis=0), mask, report)
    # (d) drop the collapsed seed? NO GT allowed for selection -- but report the
    #     oracle without-worst fusion for reference only (labelled as oracle).
    show("[oracle] mean(no s1)", q, g,
         np.mean([dists[t] for t in SEEDS if t != "s1"], axis=0), mask, report)

    with open("artifacts2/cap_ensemble5_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nrefs: old champion+TTA 0.718/0.877/0.425 | 3-seed ens 0.779/0.945/0.481")
    print("saved artifacts2/cap_ensemble5_v1.json")


if __name__ == "__main__":
    main()
