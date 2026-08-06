"""Students-only ensemble variants (CPU, saved embeddings).

    python students_only.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import numpy as np

from cowreid import CameraTopology, Manifest, build_tracklets
from cowreid.eval import EvalItem, _score
from cowreid.st_inference import INF, build_st_mask
from cowreid.tracklets import TrackletIndex
from new_levers import camera_center, dist_cosine, dist_rerank, pca_whiten, rrf

HOLD = "66.130"


def main():
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
    mask = build_st_mask(q, g, index, topo, margin=0)

    dst = np.load("_vitb_dst_emb_v3.npz", allow_pickle=True)
    ids = list(dst["ids"])
    cap = np.load("_vitb_cap_ens5_emb_v1.npz", allow_pickle=True)
    cids = list(cap["ids"]); cpos = {t: i for i, t in enumerate(cids)}
    order = [cpos[t] for t in ids]

    def champ(M):
        E = {t: M[i] for i, t in enumerate(ids)}
        Qf = np.stack([E[it.tracklet_id] for it in q]); Gf = np.stack([E[it.tracklet_id] for it in g])
        X = np.concatenate([Qf, Gf]); cc = camera_center(q + g, X)
        Qcc, Gcc = cc[:len(q)], cc[len(q):]
        Qw, Gw = pca_whiten(Gf, [Qf, Gf], n_dim=256)
        return rrf([dist_cosine(Qcc, Gcc), dist_cosine(Qw, Gw),
                    dist_rerank(Qcc, Gcc, cams_qg, k1=30, k2=6)], k=20)

    def show(n, dd):
        dm = dd.copy(); dm[mask] = INF
        r = _score(q, g, dd, (1, 5, 10)); rs = _score(q, g, dm, (1, 5, 10))
        print(f"  {n:30s}: r1={r['rank-1']:.3f} r5={r['rank-5']:.3f} mAP={r['mAP']:.3f}"
              f"  |+ST r1={rs['rank-1']:.3f} r5={rs['rank-5']:.3f} mAP={rs['mAP']:.3f}",
              flush=True)

    snames = [s for s in dst.files if s != "ids"]
    D = {s: champ(dst[s]) for s in snames}
    Dcap = {s: champ(cap[s][order]) for s in cap.files if s != "ids"}

    show("students mean(s5,s6,s7)", np.mean(list(D.values()), axis=0))
    show("s6+s7", np.mean([D[n] for n in snames if "s5" not in n], axis=0))
    show("s5+s7", np.mean([D[n] for n in snames if "s6" not in n], axis=0))
    s7 = [D[n] for n in snames if "s7" in n][0]
    show("s7 + CAP5mean 50/50", 0.5 * s7 + 0.5 * np.mean(list(Dcap.values()), axis=0))
    show("s7 x3 + CAP5 (8 terms)", np.mean([s7] * 3 + list(Dcap.values()), axis=0))
    show("students x2 + CAP5 (11 terms)",
         np.mean(list(D.values()) * 2 + list(Dcap.values()), axis=0))


if __name__ == "__main__":
    main()
