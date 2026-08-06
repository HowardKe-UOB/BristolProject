# Reproduction ladder

The fifteen scripts that produce the headline numbers, in run order. Everything else in the
repository is optional: see `experiments/`.

**Run every command from the repository root.** Data and artifact paths are relative to the
working directory, so `python repro\eval_sweep.py` is correct and `cd repro` then running it
is not. Imports keep working from any location because each script prepends the repository
root and the script folders to `sys.path` at start-up.

| # | Script | Stage | What it does |
|---|---|---|---|
| 1 | `build_phase1.py` | Data | manifest, tracklets, mined free signals, frozen splits |
| 2 | `vitb_unsup_cap.py` | Stage 1 | camera-aware proxy self-training, run with seeds 0-4 |
| 3 | `eval_cap_ensemble.py` | Stage 2 | five-seed distance-ensemble teacher |
| 4 | `make_distill_labels.py` | Stage 2 to 3 | frozen label set: intra-camera clusters + pairwise cross-camera links |
| 5 | `vitb_unsup_distill.py` | Stage 3 | holdout students (seeds 5-9) |
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
