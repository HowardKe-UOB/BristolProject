# main-results

## Zero-human operating points (mega_search_v1.json) — scalars stored are RANK-1 only (P1 r1 / dorsal-mean-r1 / P2 r1); no rank-5 or mAP breakdown exists in this file  [E:\cow626\artifacts2\mega_search_v1.json]
config | members | P1 r1 | dorsal mean r1 | P2 r1
max-P1 | dep10, m2_62, dep11, r3_100 | 0.9448 | 0.6295833333333333 | 0.6352
max-dorsal | r3_102, sup91, m2_61 | 0.8528 | 0.6903333333333332 | 0.6363
max-P2 | sup92, hc18, sup91, r3_102 | 0.8712 | 0.6737833333333333 | 0.6713

## Zero-human max-P2 / max-dorsal rerank variants (companion file)  [E:\cow626\artifacts2\mega_search2_v1.json]
metric | plain | +cluster | union variants
P2 | 0.6713 | 0.6713 | union_b0.05=0.6607, union_b0.1=0.6607, union_b0.15=0.6607
dorsal | 0.6903333333333332 | 0.6903333333333332 | union_cluster=0.6830833333333333

## Fair supervised Mega baseline (sweep_megasup_s200_v1.json) — P1 == query_66.130; headline single number is rank-1  [E:\cow626\artifacts2\sweep_megasup_s200_v1.json]
row | mAP | rank-1 | rank-5 | rank-10 | n_query
P1 (=query_66.130) | 0.8365 | 0.9693 | 0.9877 | 0.9939 | 163
query_66.1 | 0.7295 | 0.8214 | 0.875 | 0.9107 | 56
query_66.128 | 0.7824 | 0.8824 | 0.9338 | 0.9485 | 136
query_66.130 | 0.8365 | 0.9693 | 0.9877 | 0.9939 | 163
query_66.133 | 0.7642 | 0.8898 | 0.9915 | 1.0 | 118
query_66.139 | 0.5957 | 0.872 | 0.92 | 0.936 | 125
query_66.3 | 0.764 | 0.9372 | 0.9958 | 0.9958 | 239
query_66.33 | 0.7741 | 0.9245 | 0.9811 | 0.9811 | 106
dorsal_mean_r1 | - | 0.8878833333333334 | - | - | -
P2 | 0.6532 | 0.79 | 0.9321 | 0.9576 | 943

## Older DINOv2-ViT-B supervised twin (sweep_sup_full_v1.json) — byte-identical to 'SUP only (labels)' in fuse_supervised_v1.json; P1 == query_66.130  [E:\cow626\artifacts2\sweep_sup_full_v1.json]
row | mAP | rank-1 | rank-5 | rank-10 | n_query
P1 (=query_66.130) | 0.7647 | 0.8957 | 0.9755 | 0.9755 | 163
query_66.1 | 0.8026 | 0.875 | 0.9286 | 0.9464 | 56
query_66.128 | 0.85 | 0.9485 | 0.9926 | 1.0 | 136
query_66.130 | 0.7647 | 0.8957 | 0.9755 | 0.9755 | 163
query_66.133 | 0.8543 | 0.8814 | 1.0 | 1.0 | 118
query_66.139 | 0.7333 | 0.912 | 0.944 | 0.96 | 125
query_66.3 | 0.8249 | 0.9623 | 1.0 | 1.0 | 239
query_66.33 | 0.8435 | 0.9434 | 0.9906 | 0.9906 | 106
dorsal_mean_r1 | - | 0.9204333333333334 | - | - | -
P2 | 0.7196 | 0.8431 | 0.9597 | 0.983 | 943

## fuse_final_v1.json — greedy vs a-priori combos (top-level greedy_set = [sup2_92, hc18, sup2_91]). P1 fields are the query_66.130 block; dorsal = dorsal mean  [E:\cow626\artifacts2\fuse_final_v1.json]
config | P1 r1 | P1 mAP | dorsal mean | P2 r1 | P2 mAP
greedy-P2 set [sup2_92,hc18,sup2_91] | 0.8773 | 0.6304 | 0.6643833333333333 | 0.6628 | 0.45
greedy-P2 set+clust | 0.8773 | 0.6304 | 0.6643833333333333 | 0.6628 | 0.4501
sup2_trio (a-priori 3-model) | 0.8589 | 0.5765 | 0.6845666666666667 | 0.632 | 0.4204
sup2_trio+clust | 0.8589 | 0.5765 | 0.6845666666666667 | 0.632 | 0.4206
hc + all3 mega2 trios (a-priori) | 0.9018 | 0.639 | 0.6667166666666667 | 0.6437 | 0.4425
hc + all3 mega2 trios+clust | 0.9018 | 0.639 | 0.6667166666666667 | 0.6448 | 0.4428
sup2_trio + mega2ft_trio (a-priori) | 0.8712 | 0.5719 | 0.66755 | 0.6129 | 0.4192
sup2_trio + mega2ft_trio+clust | 0.8712 | 0.5719 | 0.66755 | 0.6129 | 0.4193
all19 (no-selection, all 19 models) | 0.8896 | 0.6417 | 0.6549 | 0.6405 | 0.4303
all19+clust | 0.8896 | 0.6417 | 0.6549 | 0.6405 | 0.4303

## Supplementary — SUP + zero-human blends incl. the 9-model zero-human a-priori set (fuse_supervised_v1.json). P1 = query_66.130 block; dorsal = dorsal mean  [E:\cow626\artifacts2\fuse_supervised_v1.json]
config | P1 r1 | P1 mAP | dorsal mean | P2 r1 | P2 mAP
SUP only (labels) | 0.8957 | 0.7647 | 0.9204333333333334 | 0.8431 | 0.7196
zero-human 9 (no labels) | 0.8896 | 0.6429 | 0.6587333333333333 | 0.6469 | 0.4342
SUP + zero-human 9 | 0.9325 | 0.682 | 0.6951499999999999 | 0.6808 | 0.4881
SUP x3 + zero-human 9 | 0.9387 | 0.7271 | 0.75705 | 0.7317 | 0.5667
SUP x6 + zero-human 9 | 0.9387 | 0.7536 | 0.8253333333333331 | 0.772 | 0.6326

## Supplementary — 'all-strong / no-selection' rows (NOT in mega_search_v1.json; earlier model pool hc16/mega40/hc18/megaft50). P1 = query_66.130 block  [E:\cow626\artifacts2\ensemble_search_v1.json + E:\cow626\artifacts2\final_best_v1.json]
config | P1 r1 | P1 mAP | dorsal mean | P2 r1 | P2 mAP
best4 [hc16,mega40,hc18,megaft50] (greedy) | 0.8834 | 0.6143 | 0.6334000000000001 | 0.6235 | 0.3978
best4+clust | 0.8834 | 0.6143 | 0.6334000000000001 | 0.6235 | 0.3978
all10 (no-selection) | 0.9264 | 0.6348 | 0.6177833333333334 | 0.6119 | 0.395
all10+clust | 0.9264 | 0.6348 | 0.6177833333333334 | 0.6119 | 0.395

## GOTCHAS
- HEADLINE FIELD CONVENTION: In every per-camera file, the 'P1' block == the query_66.130 query; 'dorsal_mean_r1' is the mean rank-1 over the 7 dorsal-camera queries (66.1/66.128/66.130/66.133/66.139/66.3/66.33); 'P2' is the 943-query full protocol. The single headline number for each protocol is RANK-1 (confirmed by cross-checking final_best_v1.json, where the scalar 'P1'/'dorsal'/'P2' equal the rank-1 / dorsal_mean / P2 rank-1 of the full breakdown).
- mega_search_v1.json stores ONLY single scalars (P1, dorsal, P2) per operating point — these are rank-1 / dorsal-mean-r1 / P2 rank-1. It does NOT contain rank-5 or mAP for these three sets, so the requested 'P1 r1/r5/mAP' and 'P2 r1/r5/mAP' cannot be filled from this file; only rank-1 is available. (Full breakdowns exist for OTHER sets in final_best_v1.json / ensemble_search_v1.json, but for a different model pool.)
- The 'final zero-human' headline is a COMPOSITE of three DIFFERENT ensembles, not one model/row: P1 0.9448 comes from the max-P1 set {dep10,m2_62,dep11,r3_100}; dorsal 0.6903 from the max-dorsal set {r3_102,sup91,m2_61}; P2 0.6713 from the max-P2 set {sup92,hc18,sup91,r3_102}. All three match the reference (0.945 / 0.690 / 0.671). Do not report them as one row.
- mega_search_v1.json has NO all-strong-models / no-selection row. The nearest no-selection rows are 'all10' (P1 r1 0.9264) in ensemble_search_v1.json/final_best_v1.json and 'all19' (P1 r1 0.8896) in fuse_final_v1.json — but they use different/larger model pools, so they are not a clean 'no-selection over the same candidate set' comparison against the mega_search operating points.
- Reference 'strict-holdout P1 0.883': NONE of the five target files carry a field literally named 'strict-holdout P1'. The value 0.8834 is P1 rank-1 in validate_deploy_v1.json (query_66.130 plain = 0.8834; st = 0.8773) and in sweep_final_zerohuman_v1.json (P1 rank-1 = 0.8834), and best4 P1 rank-1 in ensemble_search/final_best. So the 0.883 reference is CONSISTENT (rounds from 0.8834) and points to the deploy/final-zero-human strict-holdout eval. Beware: strict_eval_strict_v1.json reports a much lower P1 (rank-1 0.6012 plain / 0.5951 st, mAP 0.3835) — that is a harder strict transductive protocol, NOT the 0.883 figure.
- UNFAIR-COMPARISON crux: the DINOv2-ViT-B supervised twin (sweep_sup_full_v1.json) actually BEATS the fair Mega supervised baseline on dorsal (0.9204 vs 0.8879 mean r1) and on P2 rank-1 (0.8431 vs 0.79), while being LOWER on P1 rank-1 (0.8957 vs 0.9693). This inversion means the two supervised runs are almost certainly on different holdout/eval protocols — verify the protocol before placing them side by side as an apples-to-apples comparison.
- sweep_sup_full_v1.json and the 'SUP only (labels)' block of fuse_supervised_v1.json are numerically identical (P1 mAP 0.7647 / r1 0.8957, dorsal 0.9204333, P2 mAP 0.7196 / r1 0.8431) — same supervised model, no contradiction.
- The task's example 'clean a-priori combos (e.g. 9-model, 12-model)' is not literal in fuse_final_v1.json: its a-priori combos are trio-unions ('hc + all3 mega2 trios', 'sup2_trio + mega2ft_trio') plus the 19-model 'all19'. The explicit 9-model a-priori set ('zero-human 9') lives in fuse_supervised_v1.json (P1 r1 0.8896), not fuse_final_v1.json. No 12-model row appears in any of the five files. Member IDs for the 9-model set are not enumerated in the JSON.
- Reference-value check across the five target files: PASS. Final zero-human 0.9448/0.6903/0.6713 (mega_search_v1) matches 0.945/0.690/0.671; fair supervised Mega 0.9693/0.8878833/0.79 (sweep_megasup_s200) matches 0.969/0.888/0.790. No contradicting numbers found in the target files.