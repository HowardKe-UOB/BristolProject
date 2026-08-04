# ensemble-curve

## CAP per-seed strict-holdout eval s0-s4 (163 queries / 834 gallery)  [E:\cow626\artifacts2\strict_eval_cap_v1.json (s0), strict_eval_cap_s1_v1.json, strict_eval_cap_s2_v1.json, strict_eval_cap_s3_v1.json, strict_eval_cap_s4_final_v1.json (s4 used by ensemble), strict_eval_cap_s4_v1.json (superseded step-933 s4)]
seed (ckpt, step) | config | plain r1 | plain r5 | plain r10 | plain mAP | +ST r1 | +ST r5 | +ST r10 | +ST mAP
s0 (_vitb_cap_ckpt.pt, 1000) | champion recipe | 0.7117 | 0.9202 | 0.9509 | 0.4008 | 0.7117 | 0.9202 | 0.9509 | 0.4
s0 (_vitb_cap_ckpt.pt, 1000) | + flip TTA | 0.8037 | 0.9325 | 0.9693 | 0.4444 | 0.8037 | 0.9325 | 0.9693 | 0.4438
s1 (_vitb_cap_s1_ckpt.pt, 1000) | champion recipe | 0.4785 | 0.7055 | 0.816 | 0.337 | 0.4969 | 0.7117 | 0.8221 | 0.3395
s1 (_vitb_cap_s1_ckpt.pt, 1000) | + flip TTA | 0.4724 | 0.7178 | 0.816 | 0.3267 | 0.4785 | 0.7178 | 0.816 | 0.3268
s2 (_vitb_cap_s2_ckpt.pt, 982) | champion recipe | 0.681 | 0.8466 | 0.8834 | 0.4506 | 0.681 | 0.8466 | 0.8896 | 0.4521
s2 (_vitb_cap_s2_ckpt.pt, 982) | + flip TTA | 0.7239 | 0.8528 | 0.9325 | 0.4788 | 0.7239 | 0.8589 | 0.9325 | 0.4791
s3 (_vitb_cap_s3_ckpt.pt, 1000) | champion recipe | 0.6258 | 0.8098 | 0.8896 | 0.4219 | 0.6319 | 0.816 | 0.8896 | 0.425
s3 (_vitb_cap_s3_ckpt.pt, 1000) | + flip TTA | 0.6687 | 0.8405 | 0.8834 | 0.4337 | 0.6687 | 0.8466 | 0.8957 | 0.4342
s4 FINAL (_vitb_cap_s4_ckpt.pt, 1000) | champion recipe | 0.6074 | 0.8773 | 0.9509 | 0.4268 | 0.6135 | 0.8834 | 0.9509 | 0.4316
s4 FINAL (_vitb_cap_s4_ckpt.pt, 1000) | + flip TTA | 0.6871 | 0.8896 | 0.9325 | 0.4537 | 0.6933 | 0.9018 | 0.9387 | 0.4581
s4 EARLY (_vitb_cap_s4_ckpt.pt, 933) | champion recipe | 0.6687 | 0.908 | 0.9387 | 0.4062 | 0.6687 | 0.9141 | 0.9387 | 0.4087
s4 EARLY (_vitb_cap_s4_ckpt.pt, 933) | + flip TTA | 0.7055 | 0.908 | 0.9387 | 0.4387 | 0.7117 | 0.908 | 0.9448 | 0.4419

## Ensemble-size curve k=1..5, all C(5,k) subsets (+ST flip-TTA variant; no plain curve stored)  [E:\cow626\artifacts2\cap_ens_curve_v1.json]
k | n_subsets | r1 mean | r1 std | r1 min | r1 max | r5 mean | r5 std | mAP mean | mAP std
1 | 5 | 0.67362 | 0.1076 | 0.4785 | 0.8037 | 0.85152 | 0.0735 | 0.4284 | 0.0530
2 | 10 | 0.71534 | 0.0634 | 0.6196 | 0.8282 | 0.9086 | 0.0258 | 0.46094 | 0.0313
3 | 10 | 0.7282 | 0.0508 | 0.638 | 0.7914 | 0.92331 | 0.0161 | 0.47953 | 0.0153
4 | 5 | 0.73866 | 0.0348 | 0.681 | 0.7853 | 0.93252 | 0.0160 | 0.48874 | 0.0083
5 | 1 | 0.7423 | 0.0 | 0.7423 | 0.7423 | 0.9387 | 0.0 | 0.4904 | 0.0

## Consensus-weighting failure: per-seed consensus correlation vs actual solo quality  [E:\cow626\artifacts2\consensus_ens_v1.json (consensus_corr; full-precision values e.g. s0 = 0.7832365960048469, s3 = 0.8315980621807234), solo r1 from cap_ensemble5_v1.json / strict_eval_cap*_v1.json]
seed | consensus corr | consensus rank (selection order) | solo +flip-TTA r1 plain | solo +flip-TTA r1 +ST | quality rank
s0 | 0.7832 | 5th (lowest -> would be discarded first) | 0.8037 | 0.8037 | 1st (best seed)
s1 | 0.7888 | 4th | 0.4724 | 0.4785 | 5th (worst seed)
s2 | 0.8273 | 2nd | 0.7239 | 0.7239 | 2nd
s3 | 0.8316 | 1st (highest -> selection would pick s3) | 0.6687 | 0.6687 | 5th/4th tier (plain r1 0.6687)
s4 | 0.8176 | 3rd | 0.6871 | 0.6933 | 3rd

## Consensus-weighted ensemble variants (163q/834g)  [E:\cow626\artifacts2\consensus_ens_v1.json]
variant | plain r1 | plain r5 | plain r10 | plain mAP | +ST r1 | +ST r5 | +ST r10 | +ST mAP
mean(all 5) | 0.7485 | 0.9325 | 0.9571 | 0.4912 | 0.7423 | 0.9387 | 0.9632 | 0.4904
consensus-weighted mean | 0.7485 | 0.9325 | 0.9632 | 0.4917 | 0.7423 | 0.9387 | 0.9632 | 0.4909
consensus-weighted (sharp) | 0.7055 | 0.9202 | 0.9448 | 0.4966 | 0.7178 | 0.9202 | 0.9448 | 0.4972
mean(drop s0) | 0.6687 | 0.908 | 0.9448 | 0.4839 | 0.681 | 0.908 | 0.9448 | 0.4859

## Teacher 5-ensemble final rows (fusion rules over s0-s4, 163q/834g)  [E:\cow626\artifacts2\cap_ensemble5_v1.json (mean-of-5 row is numerically identical to "mean(all 5)" in consensus_ens_v1.json)]
rule | plain r1 | plain r5 | plain r10 | plain mAP | +ST r1 | +ST r5 | +ST r10 | +ST mAP
ens concat->champion | 0.7117 | 0.9325 | 0.9632 | 0.4897 | 0.7117 | 0.9325 | 0.9632 | 0.4897
ens RRF(3 champion dists) | 0.7362 | 0.9325 | 0.9693 | 0.4868 | 0.7362 | 0.9387 | 0.9693 | 0.4863
ens mean(3 champion dists) [= mean of 5, FINAL row] | 0.7485 | 0.9325 | 0.9571 | 0.4912 | 0.7423 | 0.9387 | 0.9632 | 0.4904
[oracle] mean(no s1) | 0.7853 | 0.9387 | 0.9632 | 0.502 | 0.7853 | 0.9448 | 0.9632 | 0.5014

## Mining diagnostic (link precision per teacher vs ensemble-mean, from consensus file)  [E:\cow626\artifacts2\consensus_ens_v1.json (mining_diagnostic)]
source | k=1 links | k=1 precision | k=2 links | k=2 precision
s0 | 106 | 0.5472 | 279 | 0.448
s1 | 98 | 0.4898 | 273 | 0.4139
s2 | 103 | 0.5922 | 273 | 0.4908
s3 | 96 | 0.6042 | 265 | 0.5094
s4 | 102 | 0.5882 | 271 | 0.5018
ENSEMBLE-mean | 103 | 0.6019 | 287 | 0.5192

## GOTCHAS
- Ensemble-size curve variant ambiguity resolved by arithmetic: cap_ens_curve_v1.json stores only ONE set of numbers per k, with no plain/st label. Verified k=1 mean 0.67362 = mean of the five +ST flip-TTA r1s (0.8037+0.4785+0.7239+0.6687+0.6933)/5, and k=5 point 0.7423/0.9387/0.4904 = the +ST "mean(all 5)" row. The curve is therefore the +ST variant; a plain-variant curve is NOT stored anywhere.
- Two s4 files exist: strict_eval_cap_s4_v1.json (step 933) and strict_eval_cap_s4_final_v1.json (step 1000). All ensemble artifacts (cap_ensemble5_v1.json "CAP s4", the curve, consensus file) use the step-1000 FINAL numbers (+TTA plain 0.6871/0.4537). Note the superseded step-933 checkpoint actually had HIGHER r1 (0.7055) — do not mix them.
- Headline-field ambiguity per seed: each strict_eval_cap file has 4 candidates (champion recipe vs + flip TTA, x plain/st). The ensemble pipeline consumes the "+ flip TTA" embeddings, so + flip TTA is the row consistent with fig2. For the final ensemble row, plain r1 = 0.7485 but the fig2 curve endpoint is the +ST value 0.7423 — text and figure must pick one convention or they will disagree by 0.6 pts.
- Flip TTA is not uniformly positive: it HURTS s1 (plain r1 0.4785 -> 0.4724, mAP 0.337 -> 0.3267) while helping every other seed.
- Consensus-weighting failure story, exactly as stored: selection by highest consensus correlation picks s3 (corr 0.8316, solo +TTA r1 only 0.6687), while the true best seed s0 (r1 0.8037) has the LOWEST correlation (0.7832). Sharp consensus weighting drops ensemble r1 0.7485 -> 0.7055 (plain); mean(drop s0) drops it to 0.6687. The oracle removal is s1, not s0: mean(no s1) = 0.7853 r1 (only in cap_ensemble5_v1.json, labelled [oracle]).
- Precision handling: curve means are stored with float noise (e.g. 0.7153400000000001) — reported as 0.71534 etc.; curve stds and consensus correlations are stored full-precision (e.g. rank1_std 0.10764033444764096, s0 corr 0.7832365960048469) and were rounded to 4 dp in the tables. s0 champion-recipe +ST mAP is stored literally as 0.4. The curve stores min/max for rank-1 only (no r5/mAP min/max).
- Reference-value cross-check: none of these files contain P1/dorsal/P2 splits — every entry is one protocol (n_query_scored 163, n_gallery 834, i.e. the strict-holdout eval). None of the reference numbers (0.945/0.690/0.671, 0.969/0.888/0.790, 0.883) appear, so nothing contradicts them — but note NO teacher or teacher-ensemble here reaches strict-holdout 0.883 (best single 0.8037, best non-oracle ensemble 0.7485). The 0.883 figure must come from a different artifact (final student/fusion), so do not cite these teacher numbers as the strict-holdout headline.
- File-name trap: cap_ensemble_v1.json (no "5") is an earlier 3-seed (s0,s1,s2) ensemble with different numbers (mean-of-3 r1 0.773 plain / 0.7791 st); the dissertation numbers must come from cap_ensemble5_v1.json. Also, "(3 champion dists)" in row labels refers to three distance types per member, not three seeds.
- All requested files were present; nothing missing. cap_breakdown_v1.json (checked for protocol context only) is a recipe-component breakdown for s0 and contains different, lower numbers (e.g. CAP+TTA cosine r1 0.7055) — its "champion recipe" composition differs from the strict_eval champion rows; do not source per-seed table numbers from it.