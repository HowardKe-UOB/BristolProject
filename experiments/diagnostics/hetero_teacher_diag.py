"""Diagnostic: does mining teacher links in the FUSED (DINOv2+Mega) space beat
mining in DINOv2-only space? (CPU, saved embeddings, GT for measurement only.)

All teacher pseudo-labels so far came from DINOv2 space, whose dorsal weakness
caps dorsal-link precision. Mega is dorsal-strong -> the fused space should mine
more / cleaner cross-camera links, especially dorsal-dorsal. If so, a student
retrained on fused-space labels should break the dorsal plateau at the source.

Compares, over all 7 cameras, mutual-2NN cross-camera link count + precision, and
a dorsal-only breakdown, for: DINOv2-trio, Mega-trio, and fused (mean cosine).

    python hetero_teacher_diag.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "lib" / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "lib")] + ([str(d) for d in
    (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))] if (_R / "experiments").is_dir() else [])

import json
from collections import defaultdict

import numpy as np

from cowreid import Manifest, build_tracklets
from cowreid.tracklets import TrackletIndex
from consensus_ens import mutual_knn_links

OBL = "66.130"


def prec_breakdown(links, g_tids, cam_of, gt):
    pairs = [tuple(l) for l in links]
    if not pairs:
        return 0, 0.0, 0, 0.0
    corr = sum(gt[g_tids[a]] == gt[g_tids[b]] for a, b in pairs)
    dorsal = [(a, b) for a, b in pairs
              if cam_of(g_tids[a]) != OBL and cam_of(g_tids[b]) != OBL]
    dcorr = sum(gt[g_tids[a]] == gt[g_tids[b]] for a, b in dorsal)
    return (len(pairs), corr / len(pairs),
            len(dorsal), dcorr / max(len(dorsal), 1))


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of

    dino = np.load("_vitb_dst_emb_v4.npz", allow_pickle=True)
    ids = list(dino["ids"])
    Xd = np.mean([dino[s] for s in sorted(dino.files)
                  if any(k in s for k in ("s7", "s8", "s9"))], axis=0)
    Xd = Xd / (np.linalg.norm(Xd, axis=1, keepdims=True) + 1e-12)

    mega = np.load("_sweep_mega_trio_emb.npz", allow_pickle=True)
    mids = list(mega["ids"])
    order = [mids.index(t) for t in ids]
    Xm = np.mean([mega[k][order] for k in mega.files if k != "ids"], axis=0)
    Xm = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-12)

    cams_list = [cam_of(t) for t in ids]
    Sd = (Xd @ Xd.T).astype(np.float32)
    Sm = (Xm @ Xm.T).astype(np.float32)

    report = {}
    print(f"{'space':16s} {'links':>6s} {'prec':>6s} | {'dorsal-links':>12s} "
          f"{'dorsal-prec':>11s}", flush=True)
    for name, Sfused in [("DINOv2", Sd), ("Mega", Sm),
                         ("fused 0.5/0.5", 0.5 * Sd + 0.5 * Sm),
                         ("fused 0.4/0.6", 0.4 * Sd + 0.6 * Sm)]:
        # build a fake feature by using the similarity directly in mutual_knn_links:
        # mutual_knn_links takes a feature matrix, so instead call it via a shim.
        links = mutual_knn_from_sim(Sfused, cams_list, k=2)
        n, p, dn, dp = prec_breakdown(links, ids, cam_of, gt)
        print(f"{name:16s} {n:6d} {p:6.3f} | {dn:12d} {dp:11.3f}", flush=True)
        report[name] = {"links": n, "prec": round(p, 3),
                        "dorsal_links": dn, "dorsal_prec": round(dp, 3)}

    with open("artifacts2/hetero_teacher_diag_v1.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("saved artifacts2/hetero_teacher_diag_v1.json", flush=True)


def mutual_knn_from_sim(S, cams, k=2):
    """Mutual cross-camera top-k pairs directly from a similarity matrix."""
    S = S.copy()
    n = len(S)
    cam = np.array(cams)
    for i in range(n):
        S[i, cam == cam[i]] = -2
        S[i, i] = -2
    knn = np.argsort(-S, axis=1)[:, :k]
    out = []
    for i in range(n):
        for j in knn[i]:
            if S[i, j] > -1 and i in knn[j] and i < j:
                out.append((i, int(j)))
    return out


if __name__ == "__main__":
    main()
