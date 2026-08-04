"""Build a SUPER teacher from the current strong zero-human ensemble (mega2 trio +
mega2ft trio + hc trio) and diagnose its cross-camera link precision vs the old
teachers. If dorsal-link precision beats the old 0.62, retraining students on it
is the distillation ladder's next rung (now that MegaDescriptor is in the mix).

Super teacher = weighted concat of the trio-mean spaces so inner product =
weighted avg cosine. Saved as `_super_teacher_emb.npz` (key t0) for training.

    python make_super_teacher.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cowreid import Manifest, build_tracklets
from cowreid.tracklets import TrackletIndex

OBL = "66.130"
# (npz, indices, weight) — mega2ft (dorsal-strongest) weighted most
SRC = [
    ("_sweep_mega2ft_trio_emb.npz", [0, 1, 2], 0.45),
    ("_sweep_mega2_trio_emb.npz", [0, 1, 2], 0.35),
    ("_sweep_final_zerohuman_emb.npz", [3, 4, 5], 0.20),   # hc16/17/18
]


def mutual_knn_from_sim(S, cams, k=2):
    S = S.copy(); n = len(S); cam = np.array(cams)
    for i in range(n):
        S[i, cam == cam[i]] = -2; S[i, i] = -2
    knn = np.argsort(-S, axis=1)[:, :k]
    out = []
    for i in range(n):
        for j in knn[i]:
            if S[i, j] > -1 and i in knn[j] and i < j:
                out.append((i, int(j)))
    return out


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
    np.savez_compressed("_super_teacher_emb.npz", ids=np.array(ids), t0=Xcat)
    print(f"saved _super_teacher_emb.npz shape={Xcat.shape}", flush=True)

    # link precision diagnostic
    cams_list = [cam_of(t) for t in ids]
    S = (Xcat @ Xcat.T).astype(np.float32)
    links = mutual_knn_from_sim(S, cams_list, k=2)
    pairs = [(ids[a], ids[b]) for a, b in links]
    corr = sum(gt[a] == gt[b] for a, b in pairs)
    dor = [(a, b) for a, b in pairs if cam_of(a) != OBL and cam_of(b) != OBL]
    dcorr = sum(gt[a] == gt[b] for a, b in dor)
    print(f"SUPER teacher links: {len(pairs)} @ {corr/len(pairs):.3f} | "
          f"dorsal {len(dor)} @ {dcorr/max(len(dor),1):.3f}", flush=True)
    print("refs: DINOv2 teacher dorsal-prec 0.514; fused(0.4/0.6) 0.616; Mega 0.621",
          flush=True)


if __name__ == "__main__":
    main()
