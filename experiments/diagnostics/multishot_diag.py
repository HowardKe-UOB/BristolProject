"""Temporal / multi-frame diagnostic (zero retrain): does keeping per-FRAME detail
beat pooling the 8 frames into one vector? Uses the strong mega2_s60 model.

For each eval tracklet, extract 8 per-frame backbone features (L2). Tracklet-
tracklet similarity computed 3 ways and compared to the pooled (attention) baseline
on P1 / dorsal / P2:
  pooled     : the current attention-mean single vector (baseline);
  ms-mean    : mean of all 8x8 frame-pair cosine sims (soft multi-shot);
  ms-softmax : temperature-weighted soft-max frame matching (emphasise best frames);
  ms-max     : mean over query frames of the best-matching gallery frame (best-buddy).
If any multi-shot variant beats pooled, temporal modelling is worth a retrain.

    python multishot_diag.py
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
from cowreid.encoder import DinoV2Backbone
from cowreid.eval import EvalItem, _score
from cowreid.tracklets import TrackletIndex
from st_eval_vitb import n_cls_from_ckpt
from vitb_unsup import CACHE_JSON, CACHE_NPY, CacheLoader
from vitb_unsup_mega import MegaBackbone, MegaStudent

OBL = "66.130"
CKPT = "_vitb_mega2_s60_ckpt.pt"


@torch.no_grad()
def perframe(model, cload, tids, device, bs=8):
    """{tid -> (T, D) L2-normalised per-frame backbone features}."""
    model.eval(); out = {}
    for i in range(0, len(tids), bs):
        chunk = tids[i:i + bs]
        x = cload.batch(chunk, 8, train=False)          # (b, 8, 3, H, W)
        with torch.autocast("cuda", dtype=torch.float16):
            f = model._frames(x).float()                # (b, 8, D)
        f = F.normalize(f, dim=2)
        for k, t in enumerate(chunk):
            out[t] = f[k].cpu().numpy()
    return out


def ms_dist(q_ids, g_ids, PF, mode):
    """(Nq, Ng) distance from per-frame features."""
    Q = np.stack([PF[t] for t in q_ids])                # (Nq, T, D)
    G = np.stack([PF[t] for t in g_ids])
    Nq, T, D = Q.shape; Ng = G.shape[0]
    out = np.empty((Nq, Ng))
    Gf = G.reshape(Ng * T, D)
    for i in range(Nq):
        S = Q[i] @ Gf.T                                 # (T, Ng*T)
        S = S.reshape(T, Ng, T)
        if mode == "mean":
            sim = S.mean(axis=(0, 2))
        elif mode == "max":
            s_q = S.max(axis=2).mean(axis=0)            # each q-frame best g-frame
            s_g = S.max(axis=0).mean(axis=1)
            sim = 0.5 * (s_q + s_g)
        else:                                           # softmax (tau=0.1)
            W = np.exp(S.reshape(T, Ng, T) / 0.1)
            sim = (S * W).sum(axis=(0, 2)) / (W.sum(axis=(0, 2)) + 1e-9)
        out[i] = 1.0 - sim
    return out


def pooled_dist(q_ids, g_ids, PF):
    Q = np.stack([PF[t].mean(0) for t in q_ids])
    G = np.stack([PF[t].mean(0) for t in g_ids])
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    return 1.0 - Q @ G.T


def main():
    device = "cuda"
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    by_tid = {t.tracklet_id: t for t in tracklets}
    gt = {t.tracklet_id: t.gt_label for t in tracklets}

    gal = {t.gt_label for t in tracklets if t.camera != OBL}
    ids = sorted({t.tracklet_id for t in tracklets if t.camera != OBL or t.gt_label in gal})

    ck = torch.load(CKPT, map_location="cpu")
    n_cls = n_cls_from_ckpt(ck["model"])
    backbone = MegaBackbone(pretrained=False, n_stage=1).requires_grad_(False)
    model = MegaStudent(backbone, n_cls, 256).to(device)
    model.load_state_dict(ck["model"])
    cache = np.load(CACHE_NPY, mmap_mode="r"); p2r = json.load(open(CACHE_JSON))
    cload = CacheLoader(cache, p2r, index, 8, device, np.random.default_rng(0))

    print("extracting per-frame features (mega2_s60)...", flush=True)
    PF = perframe(model, cload, ids, device)

    def run(q, g, name, report):
        line = f"  {name:8s}"
        for mode, fn in [("pooled", None), ("ms-mean", "mean"),
                         ("ms-max", "max"), ("ms-soft", "soft")]:
            qi = [it.tracklet_id for it in q]; gi = [it.tracklet_id for it in g]
            D = pooled_dist(qi, gi, PF) if fn is None else ms_dist(qi, gi, PF, fn)
            r = _score(q, g, D, (1, 5, 10))["rank-1"]
            line += f" | {mode} {r:.3f}"
            report.setdefault(name, {})[mode] = r
        print(line, flush=True)

    report = {}
    g1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != OBL]
    q1 = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
          if by_tid[t].camera == OBL and gt[t] in {x.identity for x in g1}]
    run(q1, g1, "P1", report)
    dm = defaultdict(list)
    for X in sorted({by_tid[t].camera for t in ids}):
        if X == OBL:
            continue
        g = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids if by_tid[t].camera != X]
        gs = {it.identity for it in g}
        q = [EvalItem(t, gt[t], by_tid[t].camera) for t in ids
             if by_tid[t].camera == X and gt[t] in gs]
        if q:
            run(q, g, f"q_{X}", report)
            for m in ("pooled", "ms-mean", "ms-max", "ms-soft"):
                dm[m].append(report[f"q_{X}"][m])
    print("  dorsal  " + " | ".join(f"{m} {np.mean(dm[m]):.3f}"
                                    for m in ("pooled", "ms-mean", "ms-max", "ms-soft")),
          flush=True)
    with open("artifacts2/multishot_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/multishot_diag_v1.json", flush=True)


if __name__ == "__main__":
    main()
