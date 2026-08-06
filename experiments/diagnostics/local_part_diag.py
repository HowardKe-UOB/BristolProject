"""Local-part matching, targeted at the DORSAL-DORSAL bottleneck only.

Previous part-matching failed because oblique-vs-dorsal grids don't align. But
dorsal cameras are all top-down -> their grids DO align. So test part matching
ONLY on dorsal-query-vs-dorsal-gallery pairs, fused with the global champion
distance. Uses the strong sup2 models' backbone patch tokens (Mega Swin @ 384).

For each dorsal camera as query (gallery = other DORSAL cameras only, oblique
excluded), compare:
  global   : champion distance (baseline);
  +part    : RRF(global, aligned-grid part distance) at grid 2x2 / 3x3.

    python local_part_diag.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cowreid import Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from eval_sweep import champ_dist
from st_eval_vitb import n_cls_from_ckpt
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader
from vitb_unsup_mega import MEGA_IMG, MegaBackbone, MegaStudent

OBL = "66.130"
CKPT = "_vitb_sup2_s90_ckpt.pt"


def grid_pool_tokens(tok, side, G):
    N, _, D = tok.shape
    t = tok.view(N, side, side, D)
    idx = [int(round(k * side / G)) for k in range(G + 1)]
    out = []
    for r in range(G):
        for c in range(G):
            out.append(t[:, idx[r]:idx[r + 1], idx[c]:idx[c + 1], :].reshape(N, -1, D).mean(1))
    return torch.stack(out, 1)                          # (N, G*G, D)


@torch.no_grad()
def embed_parts(model, cload, tids, device, G, bs=6):
    vit = model.backbone.model
    npre = int(getattr(vit, "num_prefix_tokens", 0) or 0)
    out = {}
    for i in range(0, len(tids), bs):
        ch = tids[i:i + bs]
        x = cload.batch(ch, 8, train=False)
        b, T = x.shape[:2]
        xf = x.flatten(0, 1)
        if xf.shape[-1] != MEGA_IMG:
            xf = F.interpolate(xf, size=(MEGA_IMG, MEGA_IMG), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = vit.forward_features(xf)              # (b*T, L, D) Swin: (b*T, H, W, D)?
        if tok.dim() == 4:
            tok = tok.flatten(1, 2)
        tok = tok[:, npre:, :].float()
        side = int(round(tok.shape[1] ** 0.5))
        parts = grid_pool_tokens(tok, side, G).view(b, T, G * G, -1).mean(1)
        parts = F.normalize(parts, dim=-1)
        for k, t in enumerate(ch):
            out[t] = parts[k].cpu().numpy()
    return out


def part_dist_aligned(q_ids, g_ids, PF):
    Q = np.stack([PF[t] for t in q_ids]); G = np.stack([PF[t] for t in g_ids])
    P = Q.shape[1]
    sim = np.zeros((len(q_ids), len(g_ids)))
    for p in range(P):
        sim += Q[:, p, :] @ G[:, p, :].T
    return 1.0 - sim / P


def rankfuse(da, db, k=60):
    def rk(d):
        o = np.argsort(d, 1, kind="stable"); r = np.empty_like(o)
        for i in range(len(d)):
            r[i, o[i]] = np.arange(d.shape[1])
        return r
    return -(1.0 / (k + rk(da)) + 1.0 / (k + rk(db)))


def main():
    device = "cuda"
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    ids = sorted({t.tracklet_id for t in tracklets})

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = MegaBackbone(pretrained=False, n_stage=1).requires_grad_(False)
    model = MegaStudent(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])
    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    # global emb (feat) for champ_dist
    d = np.load("_sweep_sup2_trio_emb.npz", allow_pickle=True)
    sids = list(d["ids"])
    gemb = [d[k] for k in d.files if k != "ids"]

    dorsal = sorted({by_tid[t].camera for t in ids} - {OBL})
    report = {}
    for G in (2, 3):
        print(f"\n=== grid {G}x{G} (dorsal-only gallery) ===", flush=True)
        PF = embed_parts(model, cload, ids, device, G)
        base_r, fused_r = [], []
        for X in dorsal:
            g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
                 if by_tid[t].camera not in (X, OBL)]         # dorsal-only gallery
            gs = {it.identity for it in g}
            q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
                 if by_tid[t].camera == X and gt[t] in gs]
            if not q:
                continue
            cams_qg = [it.camera for it in q] + [it.camera for it in g]
            gd = champ_dist(q, g, gemb, sids, cams_qg)
            qi = [it.tracklet_id for it in q]; gi = [it.tracklet_id for it in g]
            pd = part_dist_aligned(qi, gi, PF)
            rb = _score(q, g, gd, (1, 5, 10))["rank-1"]
            rf = _score(q, g, rankfuse(gd, pd), (1, 5, 10))["rank-1"]
            base_r.append(rb); fused_r.append(rf)
            print(f"  q_{X}: global {rb:.3f} -> +part {rf:.3f}", flush=True)
        print(f"  dorsal-only mean: global {np.mean(base_r):.3f} -> "
              f"+part {np.mean(fused_r):.3f}", flush=True)
        report[f"grid{G}"] = {"global": float(np.mean(base_r)),
                              "part": float(np.mean(fused_r))}
    with open("artifacts2/local_part_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/local_part_diag_v1.json", flush=True)


if __name__ == "__main__":
    main()
