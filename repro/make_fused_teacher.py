"""Build a FUSED teacher embedding whose inner product = 0.4*DINOv2 + 0.6*Mega
cosine similarity, by weighted concatenation of the two row-normalized trio-mean
spaces. Teacher structures (intra clusters + cross-camera links) mined from this
space have dorsal-link precision 0.62 vs 0.51 for DINOv2-only (hetero_teacher_diag).

Saved as `_fused_teacher_emb.npz` with a single key `t0` (2304-d), same tracklet
order as `_vitb_dst_emb_v4.npz`, for the training scripts' --teacher-npz.

    python make_fused_teacher.py
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib  # path bootstrap: keep bare-name imports working after the re-layout
_R = next(p for p in _pathlib.Path(__file__).resolve().parents if (p / "cowreid").is_dir())
_sys.path[:0] = [str(_R), str(_R / "repro"), str(_R / "common")] + [
    str(d) for d in (_R / "experiments").iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]

import numpy as np

WD, WM = 0.4, 0.6


def main():
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

    Xcat = np.concatenate([np.sqrt(WD) * Xd, np.sqrt(WM) * Xm], axis=1)
    Xcat = Xcat / (np.linalg.norm(Xcat, axis=1, keepdims=True) + 1e-12)
    np.savez_compressed("_fused_teacher_emb.npz", ids=np.array(ids), t0=Xcat)
    print(f"saved _fused_teacher_emb.npz  shape={Xcat.shape}  "
          f"(inner product = {WD}*DINOv2 + {WM}*Mega cosine)", flush=True)


if __name__ == "__main__":
    main()
