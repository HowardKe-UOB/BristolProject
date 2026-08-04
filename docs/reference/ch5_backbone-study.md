# backbone-study

## Table 1 - Frozen-backbone eval [columns: backbone | P1 rank-1 | P1 mAP | dorsal (mean r1) | P2 rank-1 | P2 mAP]  [mega_frozen_v1.json (Mega); st_inference_frozen_v1.json 'run'.'cosine' (candidate DINOv2 frozen, P1 only); SigLIP2 from project notes only; DINOv3 has no frozen file]
Mega frozen | P1 r1 0.6258 | P1 mAP 0.3961 | dorsal 0.50125 | P2 r1 0.4581 | P2 mAP 0.2751
DINOv2 frozen (backbone label INFERRED, not in JSON) | P1 r1 0.2515 | P1 mAP 0.1169 | dorsal NOT IN FILE | P2 NOT IN FILE | P2 mAP NOT IN FILE
SigLIP2 frozen | P1 0.215 | P1 mAP n/a | dorsal 0.211 | P2 0.156 | P2 mAP n/a  (PROJECT NOTES ONLY - NO JSON)
DINOv3 frozen | NO FROZEN EVAL EXISTS | - | - | - | -

## Table 2 - Trained single + trio [columns: config | P1 rank-1 | P1 mAP | dorsal (dorsal_mean_r1) | P2 rank-1 | P2 mAP]  [sweep_d3_s20_v1.json; sweep_mega_s40_v1.json; sweep_mega_trio_v1.json; sweep_mega2_s60_v1.json; sweep_mega2_trio_v1.json; sweep_mega2ft_trio_v1.json; sweep_sup2_trio_v1.json; sweep_r3_trio_v1.json]
DINOv3 s20 (trained single) | P1 r1 0.6503 | P1 mAP 0.4314 | dorsal 0.4978 | P2 r1 0.4433 | P2 mAP 0.2682
Mega n1 s40 (single) | P1 r1 0.7239 | P1 mAP 0.4690 | dorsal 0.5751 | P2 r1 0.5164 | P2 mAP 0.3196
Mega n1 trio | P1 r1 0.7178 | P1 mAP 0.4753 | dorsal 0.5561 | P2 r1 0.5090 | P2 mAP 0.3209
Mega n2 s60 (single) | P1 r1 0.8712 | P1 mAP 0.5820 | dorsal 0.59265 | P2 r1 0.5578 | P2 mAP 0.3661
Mega n2 trio | P1 r1 0.8773 | P1 mAP 0.6266 | dorsal 0.6135 | P2 r1 0.5960 | P2 mAP 0.4017
Mega2ft trio | P1 r1 0.8221 | P1 mAP 0.5427 | dorsal 0.6429 | P2 r1 0.5949 | P2 mAP 0.4031
Sup2 trio | P1 r1 0.8589 | P1 mAP 0.5765 | dorsal 0.6846 | P2 r1 0.6320 | P2 mAP 0.4204
R3 trio | P1 r1 0.8589 | P1 mAP 0.5871 | dorsal 0.6722 | P2 r1 0.6235 | P2 mAP 0.4236

## Table 3 - Heterogeneous fusion [columns: config | P1 rank-1 | P1 mAP | dorsal | P2 rank-1 | P2 mAP]  [fuse_hetero_v1.json]
dino6 (zero-human ref) | P1 r1 0.8834 | P1 mAP 0.5895 | dorsal 0.5500 | P2 r1 0.5578 | P2 mAP 0.3556
mega3 | P1 r1 0.7178 | P1 mAP 0.4753 | dorsal 0.5561 | P2 r1 0.5090 | P2 mAP 0.3209
dino6+mega3 | P1 r1 0.9264 | P1 mAP 0.6334 | dorsal 0.6123 | P2 r1 0.6161 | P2 mAP 0.3921
dino6+mega3 (mega x2) | P1 r1 0.9141 | P1 mAP 0.6285 | dorsal 0.6254 | P2 r1 0.6108 | P2 mAP 0.3958
dino6+mega3+act(clicks) | P1 r1 0.9264 | P1 mAP 0.6308 | dorsal 0.61565 | P2 r1 0.6098 | P2 mAP 0.3882
dino6+mega3 +clust | P1 r1 0.9264 | P1 mAP 0.6305 | dorsal 0.5957 | P2 r1 0.6267 | P2 mAP 0.3899
dino6+mega3+act(clicks) +clust | P1 r1 0.9264 | P1 mAP 0.6247 | dorsal 0.6022 | P2 r1 0.6246 | P2 mAP 0.3890

## Table 4 - Teacher link-precision ladder [columns: teacher | overall links | overall prec | dorsal_links | dorsal_prec]  [hetero_teacher_diag_v1.json (super-teacher 0.649/0.682 not located in any JSON)]
DINOv2 | links 352 | prec 0.560 | dorsal_links 296 | dorsal_prec 0.514
Mega | links 300 | prec 0.640 | dorsal_links 243 | dorsal_prec 0.621
fused 0.5/0.5 | links 327 | prec 0.639 | dorsal_links 266 | dorsal_prec 0.605
fused 0.4/0.6 | links 318 | prec 0.651 | dorsal_links 258 | dorsal_prec 0.616
super teacher (0.649/0.682) | NOT FOUND in any artifacts2 JSON - not stored | - | - | -

## GOTCHAS
- FIELD CONVENTION (headline mapping): in every per-config sweep JSON, the 'P1' object is identical to the 'query_66.130' object (P1 = camera 66.130 rank-1). 'dorsal' headline = 'dorsal_mean_r1' = mean rank-1 over the 6 dorsal cameras EXCLUDING 66.130 (66.1/66.128/66.133/66.139/66.3/66.33); I verified the arithmetic. 'P2' is a separate eval with n_query=943. Reported P1/P2 numbers here are rank-1 (with mAP alongside).
- mega_frozen_v1.json contains ONLY the Mega frozen backbone. It uses a slightly different schema than the sweep files (keys 'q_66.x' not 'query_66.x', 'dorsal_mean'=0.50125 not 'dorsal_mean_r1', and NO separate 66.130 entry). There is NO DINOv2 (or 'megadino') frozen entry in this file.
- DINOv2 FROZEN is ambiguous: no file gives a full P1/dorsal/P2 frozen DINOv2 eval. st_inference_frozen_v1.json ('run'.'cosine') is the best candidate for a frozen-backbone ReID eval (P1 rank-1 0.2515, mAP 0.1169) but (a) it reports P1 ONLY - no dorsal, no P2, and (b) the JSON never names the backbone; the DINOv2 identity is only inferred from the sibling file naming (st_inference_vitb_v1 / st_inference_vitb_sup_v1). Treat the DINOv2-frozen row as tentative.
- SigLIP2 frozen (0.215 / 0.211 / 0.156) has NO backing JSON anywhere in artifacts2 (grep for 'siglip' returns nothing). These three numbers exist only in project notes - flag as notes-only.
- DINOv3 has NO frozen eval - only the trained sweep_d3_s20_v1.json exists (no d3-frozen file). Confirmed.
- No 'megadino' file exists in artifacts2 (glob returned nothing).
- REFERENCE CHECK - fair supervised (Mega backbone) P1 0.969 / dorsal 0.888 / P2 0.790: CONFIRMED, maps to sweep_megasup_s200_v1.json (P1 rank-1 0.9693, dorsal_mean_r1 0.88788, P2 rank-1 0.7900). No contradiction.
- REFERENCE CHECK - final zero-human P1 0.945 / dorsal 0.690 / P2 0.671: CONFIRMED but it is a BEST-PER-AXIS COMPOSITE, not a single config. From mega_search_v1.json: max_P1 config P1=0.9448 (->0.945); max_dorsal config dorsal=0.69033 (->0.690, also mega_search2 dorsal_plain 0.6903); max_P2 config P2=0.6713 (->0.671, also mega_search2 P2_plain 0.6713). The three headline numbers come from three DIFFERENT ensemble selections - do not present them as one system unless the notes intend the per-axis best.
- REFERENCE CHECK - strict-holdout P1 0.883: COULD NOT BE REPRODUCED. strict_eval_strict_v1.json gives P1 rank-1 0.6012 plain / 0.5951 ST (+flip TTA 0.6196), and strict_eval_control_v1.json gives P1 rank-1 0.6074/0.6012 (+flip 0.6258/0.6196) - all ~0.60-0.63, not 0.883. The value 0.8834 does appear very widely but as the standard ZERO-HUMAN P1 (query_66.130 rank-1), not a strict-holdout figure. The 0.883 anchor likely comes from a different file/protocol or project notes - flag before using.
- TEACHER LADDER anchor (DINOv2 0.514 / fused 0.616 / Mega 0.621) refers to the dorsal_prec column: DINOv2 dorsal_prec 0.514, fused 0.4/0.6 dorsal_prec 0.616, Mega dorsal_prec 0.621 - CONFIRMED. Note the fused value 0.616 is the 0.4/0.6 weighting (the 0.5/0.5 fuse gives dorsal_prec 0.605). Super-teacher 0.649/0.682 is NOT stored in any artifacts2 JSON (grep found no such link-precision pair; the 0.682 hits elsewhere are unrelated rank-5/mAP values).
- Cross-file consistency verified: fuse_hetero_v1.json 'mega3' row == sweep_mega_trio_v1.json exactly; fuse_hetero_v1.json 'dino6 (zero-human ref)' row == sweep_final_zerohuman_v1.json exactly (P1 r1 0.8834, mAP 0.5895, dorsal 0.5500, P2 0.5578/0.3556). Sup2 trio and R3 trio share identical P1 rank-1 (0.8589) but differ on mAP/dorsal/P2.
- Dorsal values are reported at 4-5 decimals as stored (means carry long repeating tails, e.g. 0.5561333..., 0.6845666...); shown trimmed, none rounded up beyond the stored value. 0.50125 and 0.59265 shown at 5dp to avoid ambiguous half-rounding.