# protocol-audit

## Table 1 — Per-camera sweep, k=2 trio SSL / zero-human (0.883-era)  [E:\cow626\artifacts2\validate_protocols_v1.json]
cols: query_camera | n_query | plain rank-1 | plain mAP | st rank-1 | st mAP
full_transductive | 943 | 0.5164 | 0.3306 | 0.5164 | 0.3317
query_66.1 | 56 | 0.4821 | 0.3351 | 0.4821 | 0.3362
query_66.128 | 136 | 0.5441 | 0.3847 | 0.5441 | 0.3865
query_66.130 | 163 | 0.8773 | 0.5354 | 0.8773 | 0.5356
query_66.133 | 118 | 0.5424 | 0.3694 | 0.5424 | 0.3718
query_66.139 | 125 | 0.52 | 0.3138 | 0.52 | 0.3143
query_66.3 | 239 | 0.4184 | 0.2575 | 0.4226 | 0.261
query_66.33 | 106 | 0.5566 | 0.3922 | 0.5566 | 0.3937

## Table 2 — Per-camera sweep, SUPERVISED (cosine vs k-reciprocal rerank)  [E:\cow626\artifacts2\validate_supervised_v1.json]
cols: query_camera | n_query | cosine rank-1 | cosine mAP | rerank rank-1 | rerank mAP
full_transductive | 943 | 0.9003 | 0.7843 | 0.9555 | 0.8898
query_66.1 | 56 | 0.8929 | 0.806 | 0.9643 | 0.9562
query_66.128 | 136 | 0.9632 | 0.8778 | 0.9779 | 0.9322
query_66.130 | 163 | 0.908 | 0.7391 | 0.9632 | 0.92
query_66.133 | 118 | 0.9746 | 0.8995 | 0.9153 | 0.9502
query_66.139 | 125 | 0.944 | 0.7959 | 0.912 | 0.8573
query_66.3 | 239 | 0.954 | 0.8493 | 0.9791 | 0.9156
query_66.33 | 106 | 0.9434 | 0.8807 | 0.8774 | 0.8882

## Table 3 — Dorsal-gallery ablation (remove 66.130 from gallery) — NOT FOUND / notes-only  [none — no matching artifact in E:\cow626\artifacts2 (split_leave_out_66.130.json holds only split ID lists, no metrics)]
UNVERIFIED — no backing JSON exists. Searched all artifacts2\*.json: no 'oblique' keyword, no gallery-removal ablation file, no diag_percam_errors file.
Reference values that could NOT be confirmed: mean rank-1 0.511 -> 0.485 ; 66.33 0.557 -> 0.434 ; %rank-1 errors picking oblique 0-28%. Treat as notes-only in the dissertation unless the raw diag is regenerated.

## Table 4 — Cluster-rerank gains (base -> clustered)  [E:\cow626\artifacts2\cluster_rerank_guard_v1.json]
cols: protocol | base rank-1 | base mAP | clustered rank-1 | clustered mAP
P1 (=66.130) | 0.8957 | 0.5874 | 0.908 | 0.5967
query_66.1 | 0.5536 | 0.3703 | 0.5357 | 0.3669
query_66.128 | 0.6471 | 0.3868 | 0.6397 | 0.3882
query_66.133 | 0.5424 | 0.3622 | 0.5424 | 0.3602
query_66.139 | 0.528 | 0.3274 | 0.552 | 0.3535
query_66.3 | 0.3975 | 0.2544 | 0.4142 | 0.2572
query_66.33 | 0.5566 | 0.3942 | 0.5566 | 0.3951
P2 | 0.5536 | 0.3475 | 0.5854 | 0.361

## Table 5 — Final zero-human 6-model stack, base -> clustered  [E:\cow626\artifacts2\sweep_final_zerohuman_v1.json (base) + E:\cow626\artifacts2\cluster_rerank_finalzh_v1.json (base & clustered; base blocks identical between the two files)]
cols: protocol | base rank-1 | base mAP | clustered rank-1 | clustered mAP
P1 (=66.130) | 0.8834 | 0.5895 | 0.8957 | 0.5921
query_66.1 | 0.5357 | 0.3784 | 0.5179 | 0.3751
query_66.128 | 0.6618 | 0.3987 | 0.6471 | 0.3882
query_66.133 | 0.5593 | 0.3799 | 0.5593 | 0.3737
query_66.139 | 0.496 | 0.3289 | 0.512 | 0.3402
query_66.3 | 0.4812 | 0.2731 | 0.4812 | 0.2748
query_66.33 | 0.566 | 0.4051 | 0.5566 | 0.4018
P2 | 0.5578 | 0.3556 | 0.5748 | 0.3624
dorsal_mean_r1 (base; only in sweep_final_zerohuman_v1.json) | 0.55 | - | - | -

## Table 6 — Ensemble-search optima (source of the headline 0.945/0.690/0.671)  [E:\cow626\artifacts2\mega_search_v1.json]
cols: objective | model_set | P1 rank-1 | dorsal_mean | P2 rank-1
max_P1 | [dep10, m2_62, dep11, r3_100] | 0.9448 | 0.6295833333333333 | 0.6352
max_dorsal | [r3_102, sup91, m2_61] | 0.8528 | 0.6903333333333332 | 0.6363
max_P2 | [sup92, hc18, sup91, r3_102] | 0.8712 | 0.6737833333333333 | 0.6713

## Table 7 — Fair supervised (Mega backbone) reference confirmation  [E:\cow626\artifacts2\sweep_megasup_s200_v1.json]
cols: protocol | rank-1 | mAP
P1 (=66.130) | 0.9693 | 0.8365
dorsal_mean_r1 | 0.8878833333333334 | -
P2 | 0.79 | 0.6532

## GOTCHAS
- DORSAL-GALLERY ABLATION HAS NO ARTIFACT (Table 3). Grep across all E:\cow626\artifacts2\*.json found no 'oblique' keyword, no diag_percam_errors file, and no gallery-removal ablation. split_leave_out_66.130.json contains only train/query ID lists (no metrics). The reference numbers mean 0.511->0.485, 66.33 0.557->0.434, %oblique 0-28% are UNVERIFIABLE from files and must be flagged notes-only.
- NO diag_percam_errors JSON exists. The requested per-camera error-diagnostic artifact is absent (grep 'percam' = 0 hits).
- CLUSTER-RERANK (Table 4) CONTRADICTS the item-4 reference. cluster_rerank_guard_v1.json actually stores P1 rank-1 0.8957->0.908 and P1 mAP 0.5874->0.5967 — NOT the reference 'strict P1 0.877->0.883' nor 'mAP 0.535->0.565'. Only the P2 rank-1 gain matches: 0.5536->0.5854 (i.e. the reference '0.554->0.585'). The second P2 value '0.592' does not appear in ANY cluster_rerank file (guard P2 clustered=0.5854, finalzh=0.5748, actstack=0.6161). Report the guard file's exact values; the 0.877->0.883 / 0.535->0.565 / 0.592 figures are stale or mis-remembered.
- 'FINAL ZERO-HUMAN' IS AMBIGUOUS — two distinct things. (a) The single 6-model stack in sweep_final_zerohuman_v1.json / cluster_rerank_finalzh_v1.json gives P1 rank-1 0.8834, dorsal_mean_r1 0.55, P2 rank-1 0.5578 (base) — this is the strict-holdout-style stack, NOT 0.945. (b) The reference headline 0.945/0.690/0.671 comes from mega_search_v1.json and is THREE DIFFERENT model subsets, each optimizing a different objective (max_P1 P1=0.9448, max_dorsal dorsal=0.6903, max_P2 P2=0.6713) — not one stack. State clearly which you cite.
- REFERENCE VALUES CONFIRMED (no contradiction): strict-holdout P1 0.883 = 0.8834 (sweep_final_zerohuman_v1.json / cluster_rerank_finalzh_v1.json base P1 rank-1; also final_best_v1.json best4 and validate_deploy_v1.json 66.130 plain). Fair-supervised Mega P1 0.969 / dorsal 0.888 / P2 0.790 = sweep_megasup_s200_v1.json rank-1 0.9693 / dorsal_mean_r1 0.8878833 / P2 rank-1 0.79. k=2 trio 66.130 0.877 = validate_protocols_v1.json 0.8773. Final zero-human 0.945/0.690/0.671 = mega_search_v1.json 0.9448/0.6903/0.6713.
- HEADLINE-FIELD choice for Table 1: validate_protocols_v1.json has 'plain' and 'st' (self-training) sub-blocks that are near-identical — they differ only for query_66.3 (rank-1 0.4184 vs 0.4226) and query_66.33 (mAP 0.3922 vs 0.3937, rank-5/10 differ). 'st' is the SSL headline. For Table 2 the two blocks are 'cosine' vs 'rerank' (k-reciprocal re-ranking); rerank is the headline for full-transductive P1 (0.9555 vs 0.9003) but WARNING: per-camera rerank sometimes LOWERS rank-1 (66.33 0.9434->0.8774, 66.133 0.9746->0.9153, 66.139 0.944->0.912) — do not assume rerank always wins.
- dorsal_mean_r1 is a stored SUBSET mean, not the mean of all 7 query cameras. In sweep_final_zerohuman_v1.json it is 0.55, whereas the arithmetic mean of the 7 per-camera rank-1s is ~0.598. Report dorsal_mean_r1 as stored; do not recompute it from the 7 cameras.
- Floating-point precision: mega_search_v1.json and megasup/final_best store dorsal at full float (e.g. 0.6295833333333333, 0.8878833333333334, 0.5499999999999999). The reference rounds these (0.690, 0.888, 0.55). Use the stored full value or a clean 4-dp round; do not invent digits.
- Extra context files present if needed: validate_deploy_v1.json is a separate 'deploy' per-camera protocol (66.130 plain rank-1 0.8834 / st 0.8773, P2 st rank-1 0.5504); cluster_rerank_actstack_v1.json is a different (act-stack) rerank run with higher base P1 mAP 0.6132 and P2 rank-1 0.5864->0.6161 — do not confuse either with the guard or finalzh runs cited above.
- final_best_v1.json shows a 10-model 'all10' stack reaching P1 rank-1 0.9264 (mAP 0.6348) but dorsal_mean 0.6178 and P2 0.6119 — an alternative zero-human aggregate distinct from both the mega_search optima and the 6-model finalzh stack; mention only if the chapter needs the all-models point.