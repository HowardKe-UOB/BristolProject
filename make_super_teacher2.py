"""Super teacher v2 (ladder rung 3): sup2 trio (strongest) + mega2ft trio + hc trio,
sup2 weighted most. Diagnose link precision vs super-v1 (dorsal 0.649). If it climbs,
train rung-3 students; if flat, the ladder has converged for real.

    python make_super_teacher2.py
"""
from __future__ import annotations

import numpy as np

from cowreid import Manifest, build_tracklets
from cowreid.tracklets import TrackletIndex
from make_super_teacher import mutual_knn_from_sim

OBL = "66.130"
SRC = [
    ("_sweep_sup2_trio_emb.npz", [0, 1, 2], 0.45),
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2], 0.30),
    ("_sweep_final_zerohuman_emb.npz", [3, 4, 5], 0.25),
]


def main():
    manifest = Manifest.from_listing_file("2025Sep18.listing.txt")
    tracklets = build_tracklets(manifest, max_gap_s=2)
    index = TrackletIndex(tracklets)
    gt = {t.tracklet_id: t.gt_label for t in tracklets}
    cam_of = index.camera_of

    ids = None; parts = []
    for npz, idxs, w in SRC:
        d = np.load(npz, allow_pickle=True)
        if ids is None:
            ids = list(d["ids"])
        keys = [k for k in d.files if k != "ids"]
        trio = np.mean([d[keys[i]] for i in idxs], axis=0)
        trio = trio / (np.linalg.norm(trio, axis=1, keepdims=True) + 1e-12)
        parts.append(np.sqrt(w) * trio)
    Xcat = np.concatenate(parts, axis=1)
    Xcat = Xcat / (np.linalg.norm(Xcat, axis=1, keepdims=True) + 1e-12)
    np.savez_compressed("_super_teacher2_emb.npz", ids=np.array(ids), t0=Xcat)
    print(f"saved _super_teacher2_emb.npz shape={Xcat.shape}", flush=True)

    cams_list = [cam_of(t) for t in ids]
    S = (Xcat @ Xcat.T).astype(np.float32)
    links = mutual_knn_from_sim(S, cams_list, k=2)
    pairs = [(ids[a], ids[b]) for a, b in links]
    corr = sum(gt[a] == gt[b] for a, b in pairs)
    dor = [(a, b) for a, b in pairs if cam_of(a) != OBL and cam_of(b) != OBL]
    dcorr = sum(gt[a] == gt[b] for a, b in dor)
    print(f"SUPER-v2 links: {len(pairs)} @ {corr/len(pairs):.3f} | "
          f"dorsal {len(dor)} @ {dcorr/max(len(dor),1):.3f}", flush=True)
    print("refs: super-v1 dorsal 0.649 | fused 0.616 | DINOv2 0.514", flush=True)


if __name__ == "__main__":
    main()
