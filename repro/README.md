# Reproduction ladder

The fifteen scripts that produce the headline numbers, in run order. Everything else in the
repository is optional: see `experiments/`.

**Run every command from the repository root.** Data and artifact paths are relative to the
working directory, so `python repro\eval_sweep.py` is correct and `cd repro` then running it
is not. Imports keep working from any location because each script prepends the repository
root and the script folders to `sys.path` at start-up.

Two prerequisites are built once before Stage 1 and are not optional: the trainers `mmap` the
image cache and load the frozen ViT-S cache unconditionally, so a fresh clone must create both.

```
python common/vitb_unsup.py --build-cache --tar 2025Sep18.tar.gz   # 15 GB, CPU, ~1 h
python repro/make_vits_cache.py                                    # 9 MB, GPU, ~15 min
```

| # | Script | Stage | What it does |
|---|---|---|---|
| 0 | `make_vits_cache.py` | Prerequisite | frozen DINOv2 ViT-S features per tracklet, read by Stage 1 |
| 1 | `build_phase1.py` | Data | manifest, tracklets, mined free signals, frozen splits |
| 2 | `vitb_unsup_cap.py` | Stage 1 | camera-aware proxy self-training, run with seeds 0-4 |
| 3 | `eval_cap_ensemble.py` | Stage 2 | five-seed distance-ensemble teacher |
| 4 | `make_distill_labels.py` | Stage 2 to 3 | frozen label set: intra-camera clusters + pairwise cross-camera links |
| 5 | `vitb_unsup_distill.py` | Stage 3 | holdout students (seeds 5-9) |
| 5b | `fuse_student.py` | Stage 3 | embeds the holdout students into `_vitb_dst_emb_v4.npz`, the next rung's mining space |
| 6 | `vitb_unsup_deploy.py` | Stage 3 | deployment students (seeds 10-12) |
| 7 | `vitb_unsup_hardcl2.py` | Stage 3 | hard cannot-link students (seeds 16-18), the Stage-3 winner |
| 8 | `vitb_unsup_mega.py` | Stage 4 | MegaDescriptor students (seeds 40-42) |
| 9 | `fuse_hetero.py` | Stage 4 | heterogeneous DINOv2 + MegaDescriptor fusion |
| 10 | `fuse_final.py` | Stage 4 | final operating point, all three protocols |
| 11 | `ensemble_search.py` | Stage 4 | greedy subset search over the student pool |
| 12 | `eval_sweep.py` | Evaluation | three-protocol driver; also defines the champion distance recipe |
| 13 | `eval_ckpt.py` | Evaluation | single-checkpoint evaluation |
| 14 | `validate_protocols.py` | Evaluation | protocol audit, holdout trio |
| 15 | `validate_deploy.py` | Evaluation | protocol audit, deployment trio |

Seeds are supplied on the command line and become part of the checkpoint filename:
Stage 1 uses seeds 0-4, deployment students 10-12, hard cannot-link students 16-18, and
MegaDescriptor students 40-42. Keep these numbers: `fuse_final.py` locates each model's saved
embeddings by filename.

## Extension rungs: from 0.926 to the 0.945 selection

The core ladder above ends at the heterogeneous fusion (P1 0.926). The headline 0.945 is a
per-protocol selection over a larger model zoo, built by iterating the distillation ladder
two more rungs; convergence is declared when a further rung stops helping (Chapter 4).

| Order | Script | What it builds |
|---|---|---|
| E1 | `make_fused_teacher.py` | fused teacher: 0.4*DINOv2 + 0.6*Mega trio-mean concatenation |
| E2 | `vitb_unsup_mega.py` variants | megaft s50 (fused teacher), mega2 s60-62 (`--n-stage 2`), mega2ft s80-82 (both) |
| E3 | `make_super_teacher.py` | rung-2 super teacher (mega2ft + mega2 + hc trio-mean concat) |
| E4 | `vitb_unsup_mega.py` on it | sup2 students s90-92 |
| E5 | `make_super_teacher2.py` | rung-3 super teacher (sup2 + mega2ft + hc) |
| E6 | `vitb_unsup_mega.py` on it | rung-3 students s100-102 (this rung regresses: convergence) |
| E7 | `eval_sweep.py` per batch | one `_sweep_*_emb.npz` per trio |
| E8 | `mega_search.py` (also `ensemble_search.py`, `fuse_final.py`) | per-protocol greedy selection over the 18-model zoo; mega_search's max_P1 = 0.9448 is the 0.945 |

The exact teacher space used for the historical mega2 runs was not archived; when
reproducing, verify each retrained variant against its `sweep_*_v1.json` reference.
