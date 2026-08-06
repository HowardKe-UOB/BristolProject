# Diagnostics and audits

Measurement scripts. They train nothing; each one establishes a fact about the data, the pseudo-labels or the failure modes, and writes it to `artifacts2/`.

Run everything from the repository root, e.g. `python experiments\diagnostics\cap_breakdown.py`.

| Script | What it answers | Result archived as |
|---|---|---|
| `cap_breakdown.py` | Embeds the CAP checkpoint (feat768, normal + flip-TTA) to npz and runs the full inference-lever matrix (cosine/CC/PCAW/CC-RR/RRF, each +/- ST) against the prior champion. | cap_breakdown_v1.json |
| `cap_confirm.py` | CPU-only re-verification of the CAP 0.804 result from saved embeddings: RRF k-stability and leave-one-component-out of the champion recipe vs the prior champion. | - |
| `cap_ens_curve.py` | Ensemble-size curve for the 5 CAP seeds: evaluates every C(5,k) subset distance-mean and reports mean/std of rank-1/5 and mAP per k. | cap_ens_curve_v1.json |
| `coat_pattern_diag.py` | Samples crops from the tarball and classifies each as bicolor / mostly-dark / mostly-light, quantifying the 'fragmentary appearance' claim. | coat_pattern_diag_v1.json |
| `diag_percam_errors.py` | Explains why dorsal query cameras score low: per-camera rank-1 with full vs dorsal-only gallery, plus attribution of rank-1 errors to the oblique camera. | diag_percam_errors_v1.json |
| `distill_diag.py` | Measures pseudo-label quality in the 5-seed ensemble space (CA-Jaccard DBSCAN at several eps vs herd-size-prior agglomerative): pairwise P/R/F1 and cross-camera recall. | distill_diag_v1.json |
| `hardcl_diag.py` | Counts and profiles the hard cannot-link pairs (similarity distribution, GT violation rate, coverage) available as discrimination signal in the k=2 trio ensemble space. | - |
| `hetero_teacher_diag.py` | Compares cross-camera mutual-kNN teacher link count and precision (with dorsal-only breakdown) mined in DINOv2, Mega, and fused spaces. | hetero_teacher_diag_v1.json |
| `local_part_diag.py` | Tests aligned-grid local part matching (2x2/3x3 Swin patch-token pooling) fused with the global champion distance, restricted to dorsal-query-vs-dorsal-gallery pairs. | local_part_diag_v1.json |
| `multishot_diag.py` | Compares pooled single-vector tracklet matching against three multi-shot per-frame similarity variants (mean / softmax / best-buddy max) with no retraining. | multishot_diag_v1.json |
| `next_levers_diag.py` | Audits cross-student mutual-kNN link agreement precision and tests a cluster-consistency rerank on protocol P2. | next_levers_diag_v1.json |
| `round2_diag.py` | Compares mutual-NN cross-camera link precision in the round-2 7-model teacher space (5 CAP + 2 students) against the round-1 5-model space. | - |
| `sizecue_diag.py` | Body-size cue diagnostic: derives per-camera z-scored box area/aspect from crop file headers, measures its same-vs-different-cow AUC, then fuses it with the champion distance. | sizecue_diag_v1.json |
| `st_final_table.py` | Consolidated supervised-vs-unsupervised ViT-B comparison table, scored identically from the saved v1 embedding npz files (CPU only). | st_final_comparison_v1.json |
| `st_validate.py` | Validates the inference-time spatio-temporal mask on frozen DINOv2 features: safety (true matches removed), power (gallery pruned), and payoff (retrieval lift) at several margins. | st_mask_validation_v1.json |
| `strict_mine_diag.py` | Sweeps crop-OT mining strictness (dustbin confidence, vote count, cosine gate) in both emb256 and feat768 crop spaces and reports link count vs must-link precision. | - |
| `tune_p2.py` | CPU sweep of inference hyper-parameters (CA-Jaccard k1/k2, RRF k, ST mask) on protocol P2 for the deploy trio + hardCL-v2 student, reporting P1 alongside as a trade-off guard. | artifacts2/tune_p2_v1.json |
| `validate_supervised.py` | Runs the same two protocols on the supervised ViT-B reference embeddings to show the per-camera drop is a property of the protocol (query-camera difficulty), not of the label-free method. | artifacts2/validate_supervised_v1.json |
