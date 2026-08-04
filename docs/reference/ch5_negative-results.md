# negative-results

## GeM+BNNeck retrain (leave-out 66.130 / P1, step 1000; plain retrieval; row = config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\gembn_eval_v1.json]
cosine | 0.2592 | 0.4294 | 0.7423 | 0.7975
CC | 0.2848 | 0.4479 | 0.6871 | 0.7423
PCAW | 0.3002 | 0.5276 | 0.7669 | 0.8344
CC-RR | 0.2546 | 0.5215 | 0.6994 | 0.7362
RRF(CC,PCAW,CC-RR) [headline] | 0.3365 | 0.5276 | 0.7669 | 0.7975
RRF(CC,PCAW,CC-RR) st-variant | 0.3364 | 0.5276 | 0.7607 | 0.7975

## CC-space bootstrap pseudo-labels (leave-out 66.130 / P1, step 1600; plain; row = config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\boot_eval_v1.json]
cosine | 0.2796 | 0.5153 | 0.7362 | 0.8282
CC | 0.3141 | 0.5583 | 0.7485 | 0.8098
PCAW | 0.2223 | 0.5583 | 0.7730 | 0.8221
CC-RR | 0.3568 | 0.5337 | 0.7239 | 0.7423
RRF(CC,PCAW,CC-RR) [headline] | 0.3547 | 0.5644 | 0.7730 | 0.8528
RRF(CC,PCAW,CC-RR) st-variant | 0.3571 | 0.5644 | 0.7791 | 0.8528

## Strict high-precision mining vs over-training control (leave-out 66.130 / P1, feat768, step 1300; row = ckpt/recipe | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\strict_eval_strict_v1.json ; E:\cow626\artifacts2\strict_eval_control_v1.json]
STRICT champion recipe (plain) | 0.3835 | 0.6012 | 0.8405 | 0.9080
STRICT + flip TTA (plain) | 0.4036 | 0.6196 | 0.8650 | 0.9018
STRICT + flip TTA (st) | 0.4064 | 0.6135 | 0.8773 | 0.9018
CONTROL champion recipe (plain) | 0.3174 | 0.6074 | 0.8098 | 0.8466
CONTROL + flip TTA (plain) | 0.3207 | 0.6258 | 0.8344 | 0.8712
CONTROL + flip TTA (st) | 0.3215 | 0.6196 | 0.8344 | 0.8712

## Part matching (3x3 grid) (leave-out 66.130 / P1; plain; row = config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\part_match_v1.json]
global champion+TTA [reference] | 0.4252 | 0.7178 | 0.8773 | 0.9325
part soft | 0.1317 | 0.2025 | 0.6810 | 0.7546
part aligned | 0.1181 | 0.1902 | 0.6135 | 0.7301
part soft+TTA | 0.1458 | 0.2577 | 0.6319 | 0.7546
RRF(global+TTA, part-soft) | 0.2866 | 0.4785 | 0.8098 | 0.8957
RRF(global+TTA x2, part-soft) [best fusion] | 0.3410 | 0.5644 | 0.8712 | 0.9141
RRF(global+TTA, part-soft, part-al) | 0.2380 | 0.4049 | 0.7791 | 0.8712

## Local part matching on DORSAL-DORSAL (score = dorsal mean; row = grid | global | part)  [E:\cow626\artifacts2\local_part_diag_v1.json]
grid2 | 0.6674666666666668 | 0.6287833333333334
grid3 | 0.6674666666666668 | 0.6266

## Temporal multi-shot diagnostic (per-query rank-1; row = query | pooled | ms-mean | ms-max | ms-soft)  [E:\cow626\artifacts2\multishot_diag_v1.json]
P1 (66.130) | 0.8098 | 0.6994 | 0.7791 | 0.7607
q_66.1 | 0.5536 | 0.4821 | 0.5179 | 0.5536
q_66.128 | 0.6838 | 0.6544 | 0.7279 | 0.6471
q_66.133 | 0.5763 | 0.5000 | 0.5508 | 0.5678
q_66.139 | 0.5680 | 0.4640 | 0.5520 | 0.5040
q_66.3 | 0.4770 | 0.3808 | 0.5146 | 0.5021
q_66.33 | 0.6981 | 0.6604 | 0.7264 | 0.7264

## Hub relay v1 (holdout trio; dorsal means + P2; row = metric | value)  [E:\cow626\artifacts2\hub_relay_v1.json]
dorsal_mean_base | 0.5106
dorsal_mean_hub_only | 0.24816666666666667
dorsal_mean_rrf | 0.4471833333333333
dorsal_mean_rrf_gated@0.4 | 0.4653333333333334
dorsal_mean_rrf_gated@0.5 | 0.49645000000000006
dorsal_mean_rrf_gated@0.6 | 0.5098833333333334
dorsal_mean_qe_rrf | 0.49655
P2 base | rank-1 0.5164 | mAP 0.3306
P2 rrf_gated@0.5 | rank-1 0.5313 | mAP 0.3481
P2 rrf_gated@0.6 | rank-1 0.5323 | mAP 0.3472
P2 hub_only | rank-1 0.3277 | mAP 0.2596

## QE-hub v2 (oblique hub relay on 9-model ensemble; per-query dorsal score; row = query | base | relay@0.4 | relay@0.5 | relay@0.6)  [E:\cow626\artifacts2\qe_hub2_v1.json]
q_66.1 | 0.6786 | 0.6607 | 0.6786 | 0.6786
q_66.128 | 0.7574 | 0.7426 | 0.7647 | 0.7721
q_66.133 | 0.6102 | 0.5932 | 0.6102 | 0.6102
q_66.139 | 0.6240 | 0.6400 | 0.6320 | 0.6240
q_66.3 | 0.5314 | 0.5523 | 0.5356 | 0.5314
q_66.33 | 0.7358 | 0.7264 | 0.7264 | 0.7358
dorsal mean (computed) | base 0.6562 | relay@0.5 0.6579 | relay@0.6 0.6587 | (matches memory 0.656->0.659)

## Size cue diagnostic (log-area AUC + lambda-blend sweep; row = split/config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\sizecue_diag_v1.json]
auc_logarea | 0.49296 (near-chance 0.5, size non-discriminative)
P1 base | 0.5874 | 0.8957 | 0.9571 | 0.9755
P1 lam0.01 | 0.5649 | 0.8528 | 0.9571 | 0.9816
P1 lam0.03 | 0.4678 | 0.7730 | 0.9509 | 0.9755
P1 lam0.06 | 0.3509 | 0.6748 | 0.9080 | 0.9448
P2 base | 0.3475 | 0.5536 | 0.7105 | 0.7890
P2 lam0.06 | 0.1763 | 0.3998 | 0.5981 | 0.6766

## AQE (alpha query expansion) - UNSUP feat768 (leave-out 66.130; row = config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\st_final_comparison_v1.json]
UNSUP cosine+ST | 0.3503 | 0.6196 | 0.8344 | 0.8896
UNSUP cosine+ST+AQE | 0.3029 | 0.5399 | 0.7362 | 0.8589
UNSUP CC+ST | 0.3809 | 0.6503 | 0.8344 | 0.8773
UNSUP CC+ST+AQE | 0.3685 | 0.6196 | 0.7853 | 0.8712
SUP feat768 cosine (ref) | 0.7391 | 0.9080 | 0.9816 | 0.9877
SUP feat768 cosine+ST+AQE | 0.7811 | 0.8650 | 0.9387 | 0.9755

## DBA (database augmentation) - UNSUP feat768 (leave-out 66.130; row = config | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\new_levers_v1.json]
CC (ref, no DBA) plain | 0.3777 | 0.6564 | 0.8344 | 0.8773
CC+DBA plain | 0.3839 | 0.6258 | 0.7730 | 0.8466
CC+DBA st | 0.3870 | 0.6196 | 0.7791 | 0.8466
CC+DBA+RR plain | 0.3056 | 0.4294 | 0.5767 | 0.6810
CC+RR (ref champion component) plain | 0.4079 | 0.6626 | 0.7914 | 0.8344

## DINOv3 backbone swap - single student (row = protocol | rank-1 | mAP)  [E:\cow626\artifacts2\sweep_d3_s20_v1.json]
P1 (66.130) | 0.6503 | 0.4314
dorsal_mean_r1 | 0.49784999999999996 | (per-cam mAP not aggregated)
P2 | 0.4433 | 0.2682
reference: DINOv2 k=2 students | ~0.812 P1 | (mean 0.812 +/- 0.030)

## Distillation ladder rung-2 vs rung-3 re-mining (row = model | P1 r1 | P1 mAP | dorsal_mean_r1 | P2 r1 | P2 mAP)  [E:\cow626\artifacts2\sweep_sup2_trio_v1.json ; E:\cow626\artifacts2\sweep_r3_trio_v1.json]
rung-2 super-teacher trio (sup2) | 0.8589 | 0.5765 | 0.6845666666666667 | 0.6320 | 0.4204
rung-3 super-teacher-v2 trio (r3) | 0.8589 | 0.5871 | 0.6721666666666667 | 0.6235 | 0.4236
delta rung3-minus-rung2 | +0.0000 | +0.0106 | -0.0124 | -0.0085 | +0.0032 (converged / slightly worse)

## Consensus seed weighting - selection failure (leave-out 66.130; row = variant | mAP | rank-1 | rank-5 | rank-10)  [E:\cow626\artifacts2\consensus_ens_v1.json]
mean(all 5) plain | 0.4912 | 0.7485 | 0.9325 | 0.9571
consensus-weighted mean plain | 0.4917 | 0.7485 | 0.9325 | 0.9632
consensus-weighted (sharp) plain | 0.4966 | 0.7055 | 0.9202 | 0.9448
mean(drop s0) plain | 0.4839 | 0.6687 | 0.9080 | 0.9448
consensus_corr per seed | s0 0.7832 | s1 0.7888 | s2 0.8273 | s3 0.8316 | s4 0.8176 (s0 lowest consensus yet best actual seed -> dropping it costs -8 r1)

## hardCL single-model sweeps (row = model | P1 r1 | P1 mAP | dorsal_mean_r1 | P2 r1 | P2 mAP)  [E:\cow626\artifacts2\sweep_base_s8_v1.json ; sweep_hcl_s15_v1.json ; sweep_abl_s14_v1.json ; sweep_hcl_s13_v1.json ; sweep_hc2_s16_v1.json]
s8 baseline (CAP5 teacher, no HCL) | 0.8466 | 0.5071 | 0.4972999999999999 | 0.4889 | 0.3136
s15 (CAP5 + HCL) | 0.7730 | 0.4977 | 0.5325833333333333 | 0.4952 | 0.3230
s14 (trio teacher, no HCL) | 0.7178 | 0.4659 | 0.5098166666666667 | 0.4464 | 0.2927
s13 (trio + HCL) | 0.6503 | 0.4314 | 0.49784999999999996 | 0.4433 | 0.2682
hc2_s16 (deploy + HCL-v2) | 0.8282 | 0.5505 | 0.5209333333333334 | 0.5292 | 0.3200
paired HCL effect s8->s15 | P1 -7.4 | dorsal +3.5 | P2 +0.6

## hardCL specialist fusion (row = fusion set | P1 r1 | P1 mAP | dorsal_mean_r1 | P2 r1 | P2 mAP)  [E:\cow626\artifacts2\fuse_specialists_v1.json]
s8 only | 0.8466 | 0.5071 | 0.4972999999999999 | 0.4889 | 0.3136
s15 only | 0.7730 | 0.4977 | 0.5325833333333333 | 0.4952 | 0.3230
s8 + s15 | 0.8405 | 0.5348 | 0.5315 | 0.5207 | 0.3407
trio (baseline) | 0.8773 | 0.5354 | 0.5106 | 0.5164 | 0.3306
trio + s15 | 0.8589 | 0.5437 | 0.5251 | 0.5186 | 0.3405
trio + s15 x2 | 0.8650 | 0.5437 | 0.5380666666666667 | 0.5249 | 0.3428

## n_stage=2 Mega feasible reduced-batch (row = ckpt/stage | P1 r1 | P1 mAP | dorsal_mean_r1 | P2 r1 | P2 mAP)  [E:\cow626\artifacts2\sweep_mega2_s60_mid_v1.json ; E:\cow626\artifacts2\sweep_mega2_s60_v1.json]
mega2_s60 mid checkpoint | 0.8589 | 0.5603 | 0.5913166666666667 | 0.5429 | 0.3668
mega2_s60 final | 0.8712 | 0.5820 | 0.59265 | 0.5578 | 0.3661

## Reference-anchor verification (row = anchor | source value | task reference)  [E:\cow626\artifacts2\mega_search_v1.json ; E:\cow626\artifacts2\sweep_megasup_s200_v1.json]
Final zero-human max-P1 {dep10,m2_62,dep11,r3_100} | P1 0.9448 dorsal 0.6295833 P2 0.6352 | P1 0.945 OK
Final zero-human max-dorsal {r3_102,sup91,m2_61} | dorsal 0.6903333 P1 0.8528 P2 0.6363 | dorsal 0.690 OK
Final zero-human max-P2 {sup92,hc18,sup91,r3_102} | P2 0.6713 P1 0.8712 dorsal 0.6737833 | P2 0.671 OK
Fair supervised (Mega backbone, s200) | P1 r1 0.9693 dorsal_mean 0.8878833 P2 r1 0.7900 | P1 0.969 / dorsal 0.888 / P2 0.790 OK

## FAILED / MARGINAL FAMILIES SUMMARY (row = family | best achieved | reference at the time | delta | mechanism)  [E:\cow626\artifacts2\ (aggregated across all family JSONs above)]
GeM+BNNeck | RRF r1 0.5276 / mAP 0.3365 (st 0.5276/0.3364) | CLS champion+RRF r1 0.706 / mAP 0.423 | -18 r1 | patch-token GeM clamp discards DINOv2 negative token components; fresh BNNeck under-trained
CC-space bootstrap | RRF r1 0.5644 / mAP 0.3547 (st 0.5644/0.3571) | CLS champion 0.706/0.423 | -14 r1 | CC-space mining ~2x links but precision stayed ~15% -> 2x wrong cross-camera merges
Strict mining + over-train control | STRICT+TTA r1 0.6196 / mAP 0.4036 (st 0.4064); CONTROL+TTA r1 0.6258 / mAP 0.3207 | champion+TTA 0.718 / mAP 0.425 | -10 r1 both | clean links raise mAP (0.404 vs 0.321) but continuing self-training past step 1000 degrades model
Part matching (grid) | best fusion RRF r1 0.5644 / mAP 0.341; part-only r1 0.2577 | global champion+TTA r1 0.7178 / mAP 0.4252 | -15 r1 | fixed 3x3 grid does not align across oblique-vs-dorsal view; patch tokens weaker than CLS
Local parts on dorsal | part 0.6288 (grid2) / 0.6266 (grid3) | global 0.6675 | -3.9 to -4.1 | top-down cows face random directions so grid cells map to no fixed body part
Temporal multi-shot | best ms-max P1 0.7791; dorsal ms-max ~0.598 | pooled P1 0.8098 / dorsal 0.593 | -3.1 P1 (pooled wins) | 1fps cannot capture gait; 8 frames redundant static views, mean-pool already optimal
Hub relay v1 | dorsal gated@0.6 0.5099; P2 gated@0.6 r1 0.5323 | dorsal base 0.5106; P2 base r1 0.5164 | ~0 dorsal / +1.6 P2 | hub coverage too weak (median query->hub max-sim 0.33-0.52); rrf 0.4472 actively hurts
QE-hub v2 | dorsal relay@0.6 0.6587 | dorsal base 0.6562 | +0.3 (noise) | coverage gaps + two-hop redundant with strong 9-model ensemble
Size cue | auc_logarea 0.49296; best is base (no size) | base P1 r1 0.8957 | monotone worse (lam0.06 -> 0.6748) | log-area AUC near-chance -> body size not identity-discriminative
AQE | UNSUP cos+ST+AQE r1 0.5399 / mAP 0.3029 | UNSUP cos+ST r1 0.6196 / mAP 0.3503 | -8.0 r1 (unsup) | monochrome look-alikes pollute query expansion
DBA | CC+DBA r1 0.6258; CC+DBA+RR r1 0.4294 | CC r1 0.6564 / CC+RR r1 0.6626 | -3.1 r1 (collapse under +RR) | database augmentation pulls in look-alike neighbours
DINOv3 | single student P1 0.6503 / dorsal 0.4979 / P2 0.4433 | DINOv2 students ~0.812 P1 | -16 P1 | teacher labels mined in DINOv2 space; hypers tuned for DINOv2; DINOv3 optimised for dense tasks
Rung-3 re-mining | r3 trio dorsal 0.6722 / P2 r1 0.6235 / P1 0.8589 | rung-2 sup2 trio dorsal 0.6846 / P2 0.6320 / P1 0.8589 | dorsal -1.2, P2 -0.85 | higher link precision no longer translates; remaining errors are truly-inseparable twins (ladder converged at rung 2)
Consensus seed weighting | consensus(sharp) r1 0.7055; drop-s0 r1 0.6687 | plain mean r1 0.7485 | -4.3 to -8.0 r1 | best seed s0 has LOWEST consensus_corr (0.7832) -> weighting rewards conformity not accuracy
hardCL (marginal) | s15 dorsal 0.5326 / P2 0.4952 / P1 0.773; fusion trio+s15x2 dorsal 0.5381 / P2 0.5249 / P1 0.865 | s8 baseline dorsal 0.4973 / P2 0.4889 / P1 0.8466; trio 0.5106/0.5164/0.8773 | dorsal +3.5, P2 +0.6, P1 -7.4 | ~600 hard cannot-link pairs give fine-grained dorsal signal but too sparse to replace dense labels
n_stage=2 infeasibility | NOT stored as exact number; recorded feasible reduced-batch mega2_s60 = P1 0.8712 / dorsal 0.5927 / P2 0.5578 | n_stage=1 mega s40 dorsal 0.575 | infeasible step-45 dorsal 0.531 is memory-prose only | full-batch stage-3 unfreeze ~11 s/step (30x slower); reduced batch P=6 made it feasible

## GOTCHAS
- SOURCE-POINTER CORRECTION: the task says AQE/DBA rows are in new_levers2_v1.json, but they are NOT. new_levers2_v1.json contains only UNSUP/SUP cos/CC/PCAW/CW/CC-RR/CW-RR/RRF configs (no AQE, no DBA). The AQE rows live in st_final_comparison_v1.json (cosine+ST+AQE, CC+ST+AQE). The DBA rows live in new_levers_v1.json (feat768 CC+DBA, CC+DBA+RR). I extracted from the true source files.
- INTERNAL INCONSISTENCY on s13 (sweep_hcl_s13_v1.json): the file gives P1 0.6503 / dorsal_mean_r1 0.49785 / P2 r1 0.4433, but the project memory records s13 as .656/.529/.482. The memory's dorsal +1.9 / P2 +3.6 HCL gain for the trio-teacher pair (s14->s13) is NOT reproduced by the file (file s14->s13 = dorsal -1.2, P2 -0.3, P1 -6.8). The CAP5-teacher pair s8->s15 DOES match memory (.847/.497/.489 -> .773/.533/.495). Use the exact file numbers; treat memory's s13 dorsal/P2 as suspect. This does not touch the three task reference anchors.
- HEADLINE-FIELD choice for the CLS-token leave-out-66.130 eval files (gembn, boot, strict, part_match): the project's canonical headline is the plain-retrieval RRF(CC,PCAW,CC-RR) rank-1; the 'st' (spatio-temporal-mask) variant is reported alongside and is near-identical (+/-1 query). rank-1 is the headline metric, mAP/rank-5/rank-10 also stored.
- AQE nuance: AQE clearly HURTS the unsupervised target (feat768 cosine+ST r1 0.6196 -> 0.5399; CC+ST r1 0.6503 -> 0.6196). For SUPERVISED feat768 it is mixed (rank-1 drops 0.908 -> 0.865 but mAP rises 0.7391 -> 0.7811). The 'AQE hurt both' framing holds on rank-1 but not on supervised mAP.
- n_stage=2 'initial infeasibility': the specific infeasible step-45 result (dorsal 0.531, ~11 s/step, 45/1000 steps) is prose in memory only and is NOT stored as an exact number in any artifacts2 JSON. sweep_mega2_s60_mid_v1.json is a mid-checkpoint of the FEASIBLE reduced-batch (P=6) run (dorsal 0.5913), not the infeasible attempt. Do not cite 0.531 as an artifact-backed number.
- DINOv3 dorsal has no single aggregate mAP in sweep_d3_s20_v1.json (only dorsal_mean_r1 = 0.49785, plus per-camera mAPs). Its DINOv2 reference (~0.812 P1) comes from the k=2 distilled student trio recorded in memory, not from a sibling file in this set.
- ALL THREE task reference anchors CONFIRMED with zero contradiction: final zero-human P1 0.945 (file 0.9448), dorsal 0.690 (file 0.6903333), P2 0.671 (file 0.6713) from mega_search_v1.json; fair-supervised Mega P1 0.969 (0.9693), dorsal 0.888 (0.8878833), P2 0.790 (0.7900) from sweep_megasup_s200_v1.json. Strict-holdout P1 0.883 is the holdout trio + cluster-rerank value (memory); the raw sweep_sup2/r3 trio files show base holdout P1 rank-1 0.8589 before the +cluster-rerank +ST lift to 0.883 - so 0.883 is not directly a raw field in these specific JSONs but is consistent with them.
- QE-hub v2 (qe_hub2_v1.json) stores per-query dorsal scores only; the 0.656 base -> 0.659 relay@0.6 headline in memory is the 6-camera mean I recomputed (base 0.6562, relay@0.6 0.6587) - matches, confirming neutral result.
- Consensus-weighting failure signal: consensus_corr ranks s0 LOWEST (0.7832) but s0 is the strongest actual seed, so any consensus-based up-weighting/dropping degrades the ensemble (drop-s0 rank-1 falls 0.7485 -> 0.6687). The 'selection failure' is that no unsupervised consensus signal identifies the good seed.
- local_part_diag_v1.json values (global 0.6675 / part 0.6266-0.6288) are dorsal-dorsal aggregate scores (repeating decimals stored in full); they are a different protocol slice from the part_match_v1.json leave-out-66.130 grid experiment - do not conflate the two 'part' tables.