# Ablations and negative results

One script per claim. Each answers a single question asked in the dissertation, usually of the form *would this alternative have worked better?* Most answers are no, and those negative results are reported in Chapter 5.

Run everything from the repository root, e.g. `python experiments\ablations\cluster_rerank_guard.py`.

| Script | What it answers | Result archived as |
|---|---|---|
| `cluster_rerank_guard.py` | Applies the P2-winning cluster-consistency rerank (DBSCAN eps=0.5, bonus 0.05) to P1 and the per-camera sweep to check it generalises and never hurts. | cluster_rerank_guard_v1.json |
| `eval_phase3.py` | The no-training floors: random / colour histogram / frozen DINOv2 ViT-S (`dino`) / frozen ViT-B (`dinob`) / frozen MegaDescriptor (`mega`), the two trained backbones, on three protocols. The eight-frame reruns quoted in Chapter 5 are the `eval_phase3_*_f8_v1.json` archives. | eval_phase3*.json |
| `final_best.py` | Reports the greedy-selected 4-model set {hc16, mega40, hc18, megaft50} vs the all-10 no-selection stack, each with and without cluster rerank (10-model pool, superseded by fuse_final). | final_best_v1.json |
| `fuse_mega2.py` | Quick check of whether adding the strong n_stage=2 Mega student (mega2_s60) improves the existing zero-human fusion combos on P1/dorsal/P2. | fuse_mega2_v1.json |
| `fuse_specialists.py` | Specialist fusion negative-result test: combines the P1-strong baseline student s8 with the dorsal-strong hardCL student s15 (and the trio) across all protocols. | fuse_specialists_v1.json |
| `fuse_supervised.py` | Fuses the GT-supervised twin with the zero-human heterogeneous ensemble to measure the fusion ceiling and whether the label-free oblique strength lifts the supervised model. | fuse_supervised_v1.json |
| `hub_relay.py` | Training-free two-hop retrieval that relays dorsal-dorsal matching through oblique-camera hub tracklets, fused with the champion distance via RRF/gating/QE. | hub_relay_v1.json |
| `label_efficiency.py` | Semi-supervised label-efficiency curve: enrolls a fraction of identities with GT labels, trains the finetune+IICS pipeline multi-seed, and reports rank-1 vs labeling fraction. | - |
| `mega_frozen.py` | Zero-training evaluation of frozen MegaDescriptor-L-384 features under the champion inference recipe, as the decision gate for a full Mega retrain. | mega_frozen_v1.json |
| `mega_search.py` | Greedy forward ensemble selection over the full 21-model zero-human model zoo, per protocol (P1 / dorsal / P2), with cached per-model champion distances. | mega_search_v1.json |
| `mega_search2.py` | Pushes P2/dorsal further with greedy selection plus DBSCAN cluster-consistency reranking and weighted unions of the best sets. | mega_search2_v1.json |
| `megadino_frozen.py` | Evaluates the frozen animal-domain MegaDescriptor-DINOv2 ViT-L@518 backbone to isolate whether the s70 run collapsed from the backbone or from training divergence. | - |
| `new_levers2.py` | Round-2 label-free inference recipe: combines PCA-whitening, per-camera whitening, tuned CA-Jaccard and RRF, and checks the same recipe on supervised embeddings. | new_levers2_v1.json |
| `new_levers3.py` | Multi-backbone fusion lever: fuses the ViT-B unsupervised feature with cached ViT-S DINOv2 clip features via RRF and concatenation. | new_levers3_v1.json |
| `part_match.py` | Inference-only part/local matching from DINOv2 patch tokens (grid-pooled part descriptors, soft alignment), RRF-fused with the global champion recipe. | part_match_v1.json |
| `qe_hub2.py` | Gated oblique-camera hub-relay reranking on the strong heterogeneous ensemble: routes dorsal-dorsal comparisons through oblique tracklets, RRF-fused with the direct distance. | qe_hub2_v1.json |
| `siglip_frozen.py` | Frozen SigLIP2 so400m retrieval as a third backbone family, with flip TTA and the champion recipe, scored on all protocols. | siglip_frozen_v{N}.json (auto-versioned) |
| `strict_p1_cluster.py` | Tests a cluster-consistency rerank (same-cluster distance bonus, constrained DBSCAN over the s7/s8/s9 student mean space) on the strict P1 benchmark. | - |
| `students_only.py` | Scores students-only and student+CAP ensemble weighting variants from the v3 distilled embeddings under the champion recipe. | - |
| `students_only2.py` | Final fusion table over the v4 embeddings (5 CAP teachers + 5 distilled students), comparing student subsets and the all-10 equal-weight fusion. | final_fusion_v1.json |
| `vitb_sup.py` | Supervised twin of vitb_unsup.py: identical cache/chunk/resume machinery and budget but ground-truth inter-camera labels, giving a paired supervised reference for the inference-lever study. | artifacts2/st_inference_vitb_sup_v1.json |
| `vitb_tta.py` | Flip test-time-augmentation lever on the ViT-B unsupervised champion: averages normal and horizontally flipped embeddings, then applies the champion RRF recipe. | artifacts2/vitb_tta_v1.json |
| `vitb_unsup_active.py` | Active pairwise-labelling click-curve experiment: the system proposes its N most-confused cross-camera pairs, a simulated human answers same/different, and the answers add verified positives, hard negatives and link cleaning on top of the deployment student. | - |
| `vitb_unsup_boot.py` | Bootstrap variant that camera-centres embeddings BEFORE CA-Jaccard clustering and crop-OT mining to raise pseudo-label precision, warm-starting from the champion checkpoint (one of the runs that degraded). | artifacts2/boot_eval_v1.json |
| `vitb_unsup_dino3.py` | Backbone-swap ablation: the same k=2 distillation student recipe but on DINOv3 ViT-B/16 (inputs resized 518->512), isolating the effect of the backbone against s7/s8/s9. | - |
| `vitb_unsup_gembn.py` | Architecture ablation on the unsupervised pipeline: replaces CLS token / attention pool / 256-d projection with spatial GeM over patch tokens + temporal GeM + BNNeck, everything else unchanged. | artifacts2/gembn_eval_v1.json |
| `vitb_unsup_hardcl.py` | Route-A hard cannot-link sharpening on the holdout k=2 student: mines physically-impossible but visually similar pairs (same-camera time-overlapping, non-overlapping-camera) and pushes them apart with a hinge loss; superseded by hardcl2. | - |
| `vitb_unsup_megadino.py` | Alternative Stage-4 backbone: MegaDescriptor-DINOv2 ViT-L/14@518 (animal-domain DINOv2, no resize needed) trained with the same deployment-mode distillation recipe. | - |
| `vitb_unsup_strict.py` | Strict vs control warm-start mining experiment (superseded negative result). | - |
