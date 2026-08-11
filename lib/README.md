# lib - the library

Everything the pipeline imports lives here. The `cowreid` package holds the foundation (manifest/tracklets/topology, pair mining + OT + constrained clustering, encoder/losses/batching, splits/eval/ST-mask - it carries its own READMEs); the flat modules below are the shared runtime layer: the image cache and unsupervised trainer core, the IICS fine-tuning stack, the label-free inference levers, the pseudo-label quality diagnostic, and the spatio-temporal evaluation helpers. Several also have a CLI, but their main role is to be imported.

Run everything from the repository root, e.g. `python lib\consensus_ens.py`.

| Script | What it answers | Result archived as |
|---|---|---|
| `consensus_ens.py` | Label-free consensus seed weighting plus the cross-camera mutual-kNN link precision helper (mutual_knn_links) imported by all Stage-3/4 trainers and several diagnostics. | consensus_ens_v1.json |
| `distill_diag.py` | Measures pseudo-label quality in the 5-seed ensemble space (CA-Jaccard DBSCAN at several eps vs herd-size-prior agglomerative): pairwise P/R/F1 and cross-camera recall; exports pair_metrics imported by repro/make_distill_labels.py. | distill_diag_v1.json |
| `new_levers.py` | Library plus driver of the label-free inference levers (PCA whitening, per-camera whitening, DBA, RRF, camera centering, re-rank sweeps); its helpers are imported widely by other scripts. | new_levers_v1.json |
| `st_eval_vitb.py` | Rebuilds the ViT-B unsupervised checkpoint, embeds all eval tracklets to a new npz, and scores every label-free inference lever; also exports n_cls_from_ckpt used by other scripts. | st_inference_vitb_v1.json |
| `st_validate2.py` | Pre-screens cheap label-free inference levers (camera centering, AQE, ST mask, CA-Jaccard rerank) on frozen features; exports camera_center/aqe/run_all reused by other scripts. | st_inference_frozen_v1.json |
| `train_finetune.py` | Partial-unfreeze DINOv2 fine-tuning through raw crops (supervised / SSL, leave-out and full protocols); exports the ClipLoader class reused by later training scripts. | - |
| `train_finetune_iics.py` | Fine-tuned DINOv2 + two-stage IICS recipe (per-camera branches, inter-camera score similarity, crop-OT must-link mining); defines FineTuneIICS, train and embed_all used across the ladder. | - |
| `train_phase2.py` | Phase-2 scaffolding: build_objective() assembles the multi-task SSL loss (contrastive + Cluster-Contrast + cannot-link) plus a train_step helper; running it directly is a CPU smoke test with random weights. | - |
| `train_phase2_run.py` | Small frozen-DINOv2-feature Phase-2 run (cached features + light head, full vs loco protocols); also the home of sample_frames(), which vitb_unsup and the whole ViT-B family import. | - |
| `vitb_unsup.py` | Core unsupervised ViT-B trainer and the repo's shared infrastructure: the uint8 image cache (_imgcache.npy), CacheLoader, embed_tids/embed_crops_cached, and the CACHE/VITB/HOLD constants every ViT-B script imports. | - |
