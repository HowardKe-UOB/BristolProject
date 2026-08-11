"""Inference-only PART/LOCAL matching from DINOv2 patch tokens (no training).

Motivated by part-based pseudo-label refinement (PPLR, CVPR'22) and patch-level
mutual-NN matching for DINOv2 retrieval re-ranking: the global CLS descriptor
(feat768) is sensitive to the oblique-vs-dorsal viewpoint change; pooling the
patch tokens into a spatial grid of PART descriptors and matching parts with SOFT
alignment (each query part -> its best gallery part) is more viewpoint-robust and
is complementary to the CLS feature. We fuse a part-based distance with the
champion global recipe via reciprocal-rank fusion (RRF). Label-free, eval-only.

    python part_match.py --grid 3
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import argparse
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
from vitb_unsup import CACHE_JSON, CACHE_NPY, CKPT, HOLD, VITB, CacheLoader

RANKS = (1, 5, 10)


def grid_pool(tok, side, G):
    """(N, side*side, D) patch tokens -> (N, G*G, D) grid-mean-pooled parts."""
    N, _, D = tok.shape
    t = tok.view(N, side, side, D)
    out = []
    idx = [int(round(k * side / G)) for k in range(G + 1)]
    for r in range(G):
        for c in range(G):
            cell = t[:, idx[r]:idx[r + 1], idx[c]:idx[c + 1], :].reshape(N, -1, D)
            out.append(cell.mean(1))
    return torch.stack(out, dim=1)                      # (N, G*G, D)


@torch.no_grad()
def embed_parts(model, cload, tids, T, device, G, flip=False, bs=6):
    """{tid -> (P, D) L2-normalised part features}, temporally mean-pooled."""
    model.eval()
    vit = model.backbone.model
    npre = int(getattr(vit, "num_prefix_tokens", 1))
    out = {}
    for i in range(0, len(tids), bs):
        ch = tids[i:i + bs]
        x = cload.batch(ch, T, train=False)            # (b, T, 3, H, W)
        if flip:
            x = torch.flip(x, dims=[-1])
        b = x.shape[0]
        xf = x.flatten(0, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = vit.forward_features(xf)[:, npre:, :].float()   # (b*T, P, D)
        side = int(round(tok.shape[1] ** 0.5))
        parts = grid_pool(tok, side, G)                # (b*T, G*G, D)
        parts = parts.view(b, T, G * G, parts.shape[-1]).mean(1)  # temporal mean
        parts = F.normalize(parts, dim=-1)
        for k, t in enumerate(ch):
            out[t] = parts[k].cpu().numpy()
        del x, xf, tok, parts
    return out


def part_dist(qparts, gparts, mode="soft"):
    """(Nq,P,D),(Ng,P,D) -> (Nq,Ng) distance. soft = symmetric best-part match;
    aligned = same-index part average."""
    Q = qparts / (np.linalg.norm(qparts, axis=2, keepdims=True) + 1e-12)
    G = gparts / (np.linalg.norm(gparts, axis=2, keepdims=True) + 1e-12)
    Nq, P, D = Q.shape; Ng = G.shape[0]
    dist = np.empty((Nq, Ng), dtype=np.float64)
    for i in range(Nq):
        S = np.tensordot(Q[i], G, axes=([1], [2]))     # (P, Ng, P): S[p,n,p'] = q_p . g_{n,p'}
        if mode == "aligned":
            sim = np.mean([S[p, :, p] for p in range(P)], axis=0)
        else:                                          # soft symmetric best-match
            s_q = S.max(axis=2).mean(axis=0)           # each query part -> best gallery part
            s_g = S.max(axis=0).mean(axis=1)           # each gallery part -> best query part
            sim = 0.5 * (s_q + s_g)
        dist[i] = 1.0 - sim
    return dist


def show(name, q, g, dist, mask, report):
    r = _score(q, g, dist, RANKS); dm = dist.copy(); dm[mask] = INF
    rs = _score(q, g, dm, RANKS)
    print(f"  {name:28s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
          f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}", flush=True)
    report[name] = {"plain": r, "st": rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--frames", type=int, default=8)
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

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = DinoV2Backbone(model_name=VITB, pretrained=False).requires_grad_(False)
    model = FineTuneIICS(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])

    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, args.frames, device, np.random.default_rng(0))

    # global champion feature (feat768, CLS-based) + flip TTA
    @torch.no_grad()
    def feat768(flip=False, bs=8):
        model.eval(); out = {}
        for i in range(0, len(eval_tids), bs):
            ch = eval_tids[i:i + bs]; x = cload.batch(ch, args.frames, train=False)
            if flip:
                x = torch.flip(x, dims=[-1])
            with torch.autocast("cuda", dtype=torch.float16):
                e = F.normalize(model.head.pool(model._frames(x)), dim=1)
            for t, v in zip(ch, e.float().cpu().numpy()):
                out[t] = v
            del x
        return out

    print(f"grid={args.grid}x{args.grid}  embedding global...", flush=True)
    G0 = feat768(); G1 = feat768(flip=True)
    nrm = lambda v: v / (np.linalg.norm(v) + 1e-12)
    Gtta = {t: nrm(nrm(G0[t]) + nrm(G1[t])) for t in eval_tids}
    print("embedding parts...", flush=True)
    P0 = embed_parts(model, cload, eval_tids, args.frames, device, args.grid)
    P1 = embed_parts(model, cload, eval_tids, args.frames, device, args.grid, flip=True)

    def champ(E):
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    dG_tta = champ(Gtta)
    Qp = np.stack([P0[it.tracklet_id] for it in q]); Gp = np.stack([P0[it.tracklet_id] for it in g])
    Qpt = np.stack([0.5 * (P0[it.tracklet_id] + P1[it.tracklet_id]) for it in q])
    Gpt = np.stack([0.5 * (P0[it.tracklet_id] + P1[it.tracklet_id]) for it in g])

    report = {}
    print(f"\nleave-out {HOLD}  |Q|={len(q)} |G|={len(g)}", flush=True)
    print("-- references --")
    show("global champion+TTA", q, g, dG_tta, mask, report)
    print("-- part-only distances --")
    d_soft = part_dist(Qp, Gp, "soft"); show("part soft", q, g, d_soft, mask, report)
    d_al = part_dist(Qp, Gp, "aligned"); show("part aligned", q, g, d_al, mask, report)
    d_soft_tta = part_dist(Qpt, Gpt, "soft"); show("part soft+TTA", q, g, d_soft_tta, mask, report)
    print("-- fuse global(+TTA) champion with part --")
    show("RRF(global+TTA, part-soft)", q, g, rrf([dG_tta, d_soft_tta]), mask, report)
    show("RRF(global+TTA x2, part-soft)", q, g, rrf([dG_tta, dG_tta, d_soft_tta]), mask, report)
    show("RRF(global+TTA, part-soft, part-al)", q, g,
         rrf([dG_tta, d_soft_tta, part_dist(Qpt, Gpt, "aligned")]), mask, report)

    with open("artifacts2/part_match_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nchampion ref: global+TTA r1=0.718 r5=0.877 mAP=0.425")
    print("saved artifacts2/part_match_v1.json")


if __name__ == "__main__":
    main()
