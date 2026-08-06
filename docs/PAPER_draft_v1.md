# Label-Free Cross-Camera Cattle Re-Identification via Noise-Bounded Ensemble Distillation

*Authors: [to be filled] — draft v1, 2026-07-08*

---

## Abstract

Individual re-identification (Re-ID) of cattle across barn cameras promises continuous, per-animal welfare and productivity monitoring, but annotating identities across views is prohibitively expensive on working farms. We study fully **label-free** cross-camera cattle Re-ID in a deliberately hard, realistic setting: seven ceiling cameras with a *sparse-overlap* topology (only 7 of 21 camera pairs co-observe any area), visually near-identical animals, and a single 2.2-hour session — with no walkway ordering or coat-pattern shortcuts. We first show that the dominant failure mode of unsupervised Re-ID on such data is not representation quality but the interaction between pseudo-label **precision** and **transitivity**: cross-camera links mined at 52–60% precision collapse to 17% pairwise precision once merged transitively into clusters, a quantity we measure directly. Guided by this diagnosis we propose a **noise-bounded ensemble-distillation ladder**: (i) camera-aware-proxy self-training whose per-(cluster, camera) proxies localize the damage of wrong merges; (ii) a multi-seed *ensemble teacher* that converts high seed variance (rank-1 0.67 ± 0.11) into a cleaner neighbourhood space; and (iii) from-scratch students trained on *frozen* teacher labels with a strictly **pairwise** (never transitive) cross-camera link loss and early stopping. Students surpass their teacher (single-model rank-1 0.81 ± 0.03) and their three-model ensemble reaches **rank-1 0.883 / rank-5 0.957 / mAP 0.536** on the held-out-camera cross-view benchmark — up from 0.503 under the same protocol, with rank-5 within 1.2 points of a fully supervised twin (0.969). A per-camera protocol audit and a full-transductive evaluation qualify the scope of these gains, and a twelve-family ablation documents which standard techniques fail on monochrome livestock and why. Finally, we show that the residual dorsal-view weakness is partly a backbone-*domain* gap: heterogeneously fusing our DINOv2 students with students distilled from an off-the-shelf *animal*-Re-ID foundation model (MegaDescriptor) raises **all** protocols with zero labels — held-out-camera rank-1 to **0.926**, dorsal-query mean 0.51→**0.61**, and full-transductive to **0.616** (from a prior 0.344 label-free best). All supervision-free signals used — single-camera tracklets, timestamps, and the camera-overlap topology — are free by-products of deployment.

**Keywords:** cattle re-identification, unsupervised learning, multi-camera, pseudo-labels, ensemble distillation, precision agriculture

---

## 1. Introduction

Precision livestock farming requires knowing *which* animal is where: feeding, rumination, lameness, and estrus events are only actionable when attributed to individuals. Camera networks are cheap to install, but linking the same animal across views — cross-camera re-identification — normally requires identity annotations that working farms cannot produce at scale. Unlike humans, cattle in a single breed herd are frequently *monochrome look-alikes*; unlike the walkway settings of prior barn datasets, free-roaming barns provide no canonical camera order that would hand identity correspondence to a tracker for free.

We ask how far **fully label-free** cross-camera cattle Re-ID can be pushed on such data, and what actually limits it. Our test bed is a seven-camera barn recording of 98 cows over a single 2.2-hour session (~124k detections). Three properties make it a stress test for unsupervised Re-ID: (1) a *sparse-overlap* camera topology — only 7 of 21 camera pairs ever co-observe an area — so cross-view correspondence cannot be brute-forced from co-occurrence; (2) near-identical animal appearance, which poisons appearance-only pseudo-labelling; and (3) an oblique-view camera whose viewpoint differs drastically from the six dorsal views, which we hold out entirely from training and use as the query side of our primary benchmark.

Our starting point reproduces the community's standard unsupervised pipeline — DINOv2 features fine-tuned with cluster-contrast self-training on tracklet pseudo-labels — at rank-1 0.503 on this benchmark, against a supervised twin at 0.963. Closing this gap proceeds in two acts.

**Act one: measurement before modelling.** Retrieval-side corrections that require no training — reading out the pre-projection backbone feature, per-camera centering, PCA whitening, camera-aware re-ranking, reciprocal rank fusion, and flip test-time augmentation — already lift rank-1 to 0.718. More importantly, instrumenting the training pipeline reveals two quantitative facts that drive everything after. First, *seed variance dominates*: the same self-training recipe spans rank-1 0.47–0.80 across five random seeds, one in five collapses outright, and no label-free signal we tested (cluster-count trajectories, cross-seed consensus) can identify the bad runs — consensus weighting in fact discards the *best* seed. Second, *transitivity destroys precision*: cross-camera links mined in a strong feature space at 52–60% precision fall to **17%** pairwise precision after standard union-find merging into clusters, because a single wrong link fuses whole groups. This measurement explains, in one number, why our three prior attacks on the pseudo-label bottleneck (an architecture swap, clustering-space bootstrapping, strict high-precision mining) all failed.

**Act two: a ladder where every rung bounds noise.** We propose an ensemble-distillation procedure designed around those two facts. Camera-aware proxy (CAP) self-training replaces cluster centroids with per-(cluster, camera) proxies, so a wrong merge corrupts one proxy rather than a whole identity prototype. Five CAP seeds are combined into a distance-mean *ensemble teacher* — converting seed variance from a liability into signal (rank-1 0.749, with a monotone mean-up/variance-down ensemble-size curve over all seed subsets). In the teacher space we mine intra-camera clusters at 0.879 pairwise precision and cross-camera mutual k-NN links at 52% precision, then train students from scratch with those labels **frozen** and with the links applied **pairwise** — as individual confidence-weighted proxy attractions, never as cluster merges. Students surpass the teacher at one fifth of its inference cost (a single model vs. five) and with 3.5× lower seed variance (0.81 ± 0.03), and their three-model ensemble reaches **rank-1 0.883 / rank-5 0.957 / mAP 0.536**.

We report what does *not* transfer with equal prominence. A per-camera audit shows the headline is protocol-specific: with dorsal cameras as queries the same models score 0.42–0.56, while a supervised twin is uniformly strong (0.88–0.98) — the remaining gap is methodological headroom, not protocol difficulty. On the full-transductive protocol our system reaches 0.549 (with deployment-mode training that mines links across all seven cameras), a 20-point improvement over the prior label-free best on this data but far from the supervised 0.956.

**Contributions.**
1. **A measured failure law and a design rule.** We quantify the precision–transitivity collapse of mined cross-camera links (60% → 17% under union-find merging) and derive the rule that makes moderate-precision links usable: apply them pairwise, never merge (§4.4, §5.4).
2. **A noise-bounded ensemble-distillation ladder** for unsupervised Re-ID: CAP self-training → multi-seed distance-ensemble teacher → frozen-label, pairwise-link, early-stopped students. Each stage is chosen to bound the damage of label noise; students beat their teacher and collapse seed variance (§4, §5.3–5.4).
3. **Topology-as-supervision for free-roaming barns.** Same-instant detections in non-overlapping cameras are different animals with 99.9% reliability (~625k cannot-link pairs), inverting the positive spatio-temporal priors of person Re-ID into the negative signal that sparse-overlap livestock settings actually support (§4.1).
4. **An honest empirical map.** A twelve-family ablation with mechanisms for every failure (over-training, clustering-space centering, patch-token pooling, consensus seed selection, query expansion), a per-camera protocol audit, an all-subset ensemble-size curve, and a label-efficiency curve — establishing both the new label-free state of this dataset (0.883 held-out-camera / 0.549 full-transductive) and its current limits (§5.5–5.7).

## 2. Related Work

**Unsupervised person Re-ID.** Modern unsupervised Re-ID alternates clustering-based pseudo-labelling with contrastive or classification training: cluster-contrast memory banks [ClusterContrast], self-paced hybrid memories [SpCL], and tracklet-association formulations that use tracker output as free intra-camera supervision [TAUDL]. Camera-aware variants recognize that camera bias corrupts both distances and clusters: IICS alternates intra- and inter-camera similarity learning [IICS]; CAP and O2CAP attach proxies to (cluster, camera) pairs [CAP, O2CAP]; CA-Jaccard makes k-reciprocal neighbourhoods camera-aware [CAJaccard]; PPLR refines labels with part-global agreement [PPLR]. We adopt the CAP idea as our first rung and show that on sparse-overlap livestock data its proxy-level losses are not merely a refinement but a survival requirement — centroid-level cluster-contrast collapses under 15%-precision merges. Distinct from all of the above, our second rung freezes labels entirely and moves the cross-camera signal out of the cluster structure into pairwise losses, motivated by a transitivity-collapse measurement we have not seen articulated in this literature.

**Spatio-temporal priors.** TFusion and ST-ReID fuse camera transition-time distributions with appearance scores [TFusion, STReID]. These are *positive* priors suited to corridor-like human topologies. In a free-roaming barn transition times are diffuse; what is reliable is the *negative* signal — an animal cannot be in two non-overlapping fields of view at the same instant — which we use as dense cannot-link supervision during clustering and training, and (optionally) as an impossibility mask at retrieval.

**Animal and cattle Re-ID.** Coat-pattern biometrics work well for Holstein-Friesian herds [MultiCamCows], and walkway or milking-parlour deployments obtain cross-camera correspondence from enforced animal ordering. MultiCamCows2024 reports >96% self-supervised accuracy by exploiting exactly that sequential structure [MultiCamCows]. Our setting removes both crutches — monochrome animals, unordered movement, sparse overlap — and our results quantify how much harder the problem then becomes, and how much of the gap disciplined label-noise management can recover.

**Ensembles, distillation and retrieval post-processing.** Our teacher aggregates independently trained seeds by averaging distance matrices, and students distil it through mined constraints rather than logits, in the spirit of self-distillation amplification [SimCLR-style read-out choices; BNNeck]. The inference stack composes known tools — pre-projection read-out [BNNeck], PCA whitening, k-reciprocal and camera-aware re-ranking [kReciprocal, CAJaccard], reciprocal rank fusion [RRF], flip TTA — whose *combination* we tune once and reuse unchanged across all models and protocols.

## 3. Problem Setting

**Data.** `2025Sep18`: seven ceiling-mounted cameras in a free-stall barn, one afternoon session of ~2.2 h at ~1 fps, 98 identities, 124,145 body-crop detections with timestamps. Detections are grouped into 997 single-camera **tracklets** (max gap 2 s) — the standard free supervision assumed available from any multi-object tracker [TAUDL]. Ground-truth identities are used exclusively for evaluation and for *measuring* (never selecting with) pseudo-label quality.

**Camera topology.** The cameras do not share one field of view: only 7 of 21 pairs ever co-observe an area. Consequently only 1.66% of same-instant cross-camera crop pairs share an identity — same-time co-occurrence is nearly useless as a positive signal but its inverse is excellent: in *non-overlapping* pairs, same-instant detections are different animals with 99.86% reliability, yielding ~6.25 × 10⁵ cannot-link pairs.

**Protocols.** (P1) **Held-out-camera cross-view** (primary): the oblique camera 66.130 is excluded from all training; its tracklets form the query set (n = 163) against the six dorsal cameras as gallery (n = 834), Market-1501 junk rule [Market1501]. P1 models a practical deployment task — *new-camera onboarding*: a newly installed camera, with a different viewpoint and zero annotation, must be linked into an existing camera network. This benchmark was fixed before method development and is the hardest viewpoint change in the network. (P2) **Full transductive**: query = gallery = all 943 tracklets whose identity appears in ≥2 cameras. (P3) **Per-camera sweep**: each camera in turn as query against all others. Metrics: CMC rank-1/rank-5 and mAP; with n = 163, one query ≈ 0.6 rank-1 points.

**Supervised reference.** A twin of our architecture trained with ground-truth identities (same backbone, budget and schedule) reaches 0.963/0.969/0.920 on P1 and 0.956 rank-1 on P2, providing the ceiling all label-free numbers are compared against.

## 4. Method

Figure 6 gives the overview; Figure 7 illustrates the three mechanisms. The system has four stages: free input structure (§3), CAP self-training (§4.2), the ensemble teacher with its mining diagnostics (§4.3), pairwise-link distillation (§4.4), and a label-free inference stack (§4.5).

### 4.1 Topology and temporal signals

From timestamps and the (label-free estimable) overlap topology we derive: (i) **cannot-link constraints** — same-instant tracklet pairs in non-overlapping cameras — injected into every clustering step and as a repulsion loss during training; (ii) **mining bags** — same-instant crop sets on *overlapping* pairs, the only place where cross-view positives are physically possible; and (iii) an optional retrieval-time **impossibility mask** (distance → ∞ for time-overlapping tracklets in non-overlapping cameras), which we show is safe (0.37% true-match loss) but low-powered on P1 (§5.8).

### 4.2 Stage 1: camera-aware proxy self-training

Each seed trains DINOv2 ViT-B/14 (last four blocks unfrozen) with two alternating objectives. *Intra-camera:* per-camera pseudo-identities from constrained clustering feed per-camera cosine classifiers. *Inter-camera:* CA-Jaccard DBSCAN clusters (cannot-link aware) are augmented by crop-level optimal-transport mining on overlapping pairs — a signal we measure at only ~15% precision — and represented by **per-(cluster, camera) proxies** with momentum updates [CAP]. The loss pulls an instance to its own proxy against same-camera proxies (softmax CE) and softly toward same-cluster different-camera proxies (contrast over all different-camera proxies). The design point (Fig. 7a): a wrong cross-camera merge contaminates *one proxy pair*, not a global centroid — under 15%-precision mining this containment is what keeps training alive where cluster-contrast variants degrade.

### 4.3 Stage 2a: the multi-seed ensemble teacher

Five independent seeds yield rank-1 0.671 ± 0.110 — and, decisively, no label-free criterion identifies the collapsed run: cluster-count trajectories of good and bad seeds are indistinguishable, and leave-one-out consensus correlation ranks the *best* seed as the outlier (dropping it costs 8 points). Instead of selecting, we **average the seeds' distance matrices**. Over all C(5,k) seed subsets the ensemble's mean rises monotonically (0.674 → 0.742) while its spread shrinks to zero (Fig. 2) — a selection-free, reproducible teacher at rank-1 0.749 whose *neighbourhood structure* is cleaner than any member's.

That structure is the point. In teacher space: intra-camera clusters reach 0.879 pairwise precision; cross-camera **mutual k-NN tracklet links** reach 60.2% precision at k = 1 (103 links) and 51.9% at k = 2 (287 links) — 3.5× the precision of the in-training crop-OT signal. Full *clusterings* of the same space, by contrast, never exceed ~21% cross-camera pair precision regardless of algorithm or granularity: transitive grouping, not the feature space, is what destroys precision.

### 4.4 Stage 2b: pairwise-link distillation

We make that observation explicit. Merging the 60%-precision k = 1 links into the intra-camera clusters by union-find yields merged groups whose cross-camera pair precision is **16.7%** — a single wrong link fuses entire clusters and manufactures hundreds of wrong pairs (Fig. 7b). Therefore students never merge. A student is trained from scratch with: (i) the teacher's intra-camera clusters, **frozen** for the entire run, as per-camera classifiers and proxies (no refresh ⇒ no drift, no collapse — the over-training failure of §5.5 cannot occur); (ii) for every mined link (a, b), a **pairwise** loss pulling instances of proxy *a* toward proxy *b* against all different-camera proxies, weighted by the link's cosine confidence — each wrong link mis-attracts exactly one proxy pair, bounding its damage at source; (iii) early stopping at 1000 steps. Batches oversample linked proxies so the sparse links fire.

Students inherit the teacher's knowledge in a cleaner form than the teacher itself holds it: k = 2 students score 0.773/0.847/0.816 across three seeds (0.812 ± 0.031 — each *single model* above every previous ensemble), and recall beats precision once damage is bounded (k = 2's 287@52% links outperform k = 1's 103@60%: 0.812 vs 0.712 mean). The ladder converges after one rung — re-mining in the improved student-ensemble space does not raise link precision (0.591 vs 0.602) — so we stop at one distillation round and ensemble the three students by distance averaging.

### 4.5 Label-free inference stack

Per student: pre-projection 768-d read-out [BNNeck]; horizontal-flip TTA; then three complementary views of the gallery — per-camera centering, gallery-fit PCA whitening (256-d), and camera-aware k-reciprocal re-ranking on the centered feature [CAJaccard, kReciprocal] — fused by reciprocal rank fusion (k = 20) [RRF]. Student distances are averaged; the §4.1 impossibility mask is optional. Every component is label-free and fixed once across all experiments.

## 5. Experiments

### 5.1 Implementation

PyTorch, AMP, one consumer GPU (RTX 5090 laptop). 518² crops, 8 frames per tracklet, attention temporal pooling; AdamW (backbone 1e-5, head 3e-4, wd 1e-4); P×K sampling (10–12 × 4); 1000 steps per model with cluster/mining refresh every 250 (Stage 1) or no refresh (students). A uint8 image cache and 240-s checkpoint-resume chunks make every run restartable — training survived multiple machine freezes unchanged. Total budget: ~8 model trainings × ~40 GPU-minutes.

### 5.2 Main result (P1)

Table 1 / Figure 1: the journey decomposes into training-free protocol corrections (0.503 → 0.718) and the two-stage ladder (→ 0.749 → **0.883**). Rank-5 0.957 approaches the supervised twin's 0.969; the rank-1 gap (0.883 vs 0.963) remains.

**Table 1 — Held-out-camera cross-view (P1), label-free.**

| Configuration | R1 | R5 | mAP |
|---|---|---|---|
| Cluster-contrast baseline, 256-d projection read-out | 0.503 | 0.736 | 0.330 |
| + 768-d backbone read-out | 0.620 | 0.841 | 0.347 |
| + camera centering + CA-Jaccard re-rank | 0.663 | 0.791 | 0.408 |
| + reciprocal rank fusion (3 views) | 0.706 | 0.859 | 0.423 |
| + flip TTA | 0.718 | 0.877 | 0.425 |
| CAP 5-seed ensemble (Stage 1+2a) | 0.749 | 0.932 | 0.491 |
| **Ensemble-distilled trio, k=2 (full ladder)** | **0.883** | **0.957** | **0.536** |
| *Supervised twin (+re-rank)* | *0.963* | *0.969* | *0.920* |

### 5.3 Seed variance and the ensemble teacher

Per-seed CAP results span 0.472–0.804 (Table 2). The all-subset ensemble-size curve (Fig. 2) is monotone in mean and variance; we highlight that an earlier 3-seed result (0.779) sat in the lucky tail of the k = 3 distribution (0.728 ± 0.051) — single-subset ensemble numbers mislead exactly like single seeds do. Consensus-based seed weighting fails structurally: the strongest seed disagrees most with the mediocre majority.

**Table 2 — CAP seeds (champion recipe + TTA) and students.**

| Model | R1 | R5 | mAP |
|---|---|---|---|
| CAP seeds s0–s4 | .804 / .472 / .724 / .669 / .687 | — | — |
| CAP single-seed mean | 0.671 ± 0.110 | 0.847 ± 0.072 | 0.428 ± 0.052 |
| CAP 5-seed ensemble | 0.749 | 0.932 | 0.491 |
| k=1 students (2 seeds) | 0.712 / 0.712 | — | — |
| k=2 students (3 seeds) | 0.773 / 0.847 / 0.816 | — | — |
| k=2 single-model mean | **0.812 ± 0.031** | 0.945 ± 0.015 | 0.502 ± 0.015 |
| k=2 trio ensemble | **0.883** | **0.957** | **0.536** |

### 5.4 Why pairwise links work

Link mining in teacher space: k = 1/2/3/4 → 103/287/491/722 links at 60.2/51.9/44.8/40.9% precision. Transitive union-find merging of the k = 1 set collapses cross-camera pair precision to 16.7%; every full clustering of the same space stays ≤ ~21%. Used pairwise, k = 2 beats k = 1 by 10 rank-1 points of student quality — with bounded damage, an extra 184 links at 52% outweigh 60% precision on 103. Students are also *low-variance* (± 0.031 vs ± 0.110): frozen labels remove the self-training lottery.

### 5.5 What did not work (and why)

Twelve method families were evaluated under identical protocols (Fig. 5; deltas are measured against the best label-free result at the time of each experiment, i.e. 0.706 pre-TTA or 0.718 post-TTA). Failures with mechanisms: **GeM+BNNeck architecture swap** (−17.8): clamping patch tokens for fractional powers discards DINOv2's negative components; on monochrome animals GeM amplifies background peaks while the CLS token is already the strong holistic descriptor. **Clustering-space centering bootstrap** (−14.2): per-camera centering aids *ranking* but removes the within-camera discriminative offsets that keep look-alike animals apart, doubling links at unchanged ~13% precision. **Continued self-training** (−9.8): both loose and strict re-mining regress equally when training continues past the early stop — over-training on accumulating pseudo-label noise, not mining quality, dominates (the strict arm's much higher mAP, 0.404 vs 0.321, confirms precision is a real but subordinate knob). **Part matching** (−15.4): grid parts do not align across oblique-vs-dorsal viewpoints. **Frozen-backbone multi-model fusion** (−2.5), **AQE** (−8.0) and **DBA**: weak or contaminated neighbours poison expansions among look-alikes.

### 5.6 Protocol robustness (P2, P3)

Per-camera audit (Fig. 3, Table 3): with the unseen oblique camera as query the trio scores 0.877–0.883; with the six *seen* dorsal cameras as queries only 0.418–0.557; full-transductive 0.516. The supervised twin is uniform (0.877–0.979; P2 0.956) — the asymmetry is a property of our method, not the protocols. A gallery-ablation diagnostic localizes the deficit: removing the oblique camera from the gallery does *not* recover dorsal-query performance (mean rank-1 0.511 → 0.485; only 0–28% of rank-1 errors select an oblique item, and one camera *drops* 12 points because its easiest true matches lived in the oblique view). The weakness is intrinsic to **dorsal–dorsal** matching, where top-down views of monochrome animals offer few discriminative cues — exactly the look-alike pairs whose separation, in the supervised twin, is taught by identity labels. Oblique queries, by contrast, carry richer body-side cues and retrieve *into* the mutually-aligned dorsal gallery, which is why the unseen camera is paradoxically the easiest query side. Deployment-mode training (teacher links re-mined across all seven cameras: 352 links @ 56%, 56 involving the oblique camera) lifts P2 to **0.549** and most seen-camera queries (e.g. 66.128 +8.1), the best label-free full-transductive result on this data (prior best 0.344), while the asymmetry persists — the central open problem this paper leaves.

**Table 3 — Per-camera rank-1 (P3) and full transductive (P2).**

| Query camera | 66.1 | 66.128 | 66.130* | 66.133 | 66.139 | 66.3 | 66.33 | P2 |
|---|---|---|---|---|---|---|---|---|
| Label-free (holdout mode) | .482 | .544 | **.877** | .542 | .520 | .418 | .557 | .516 |
| Label-free (deployment mode) | .554 | .625 | .883 | .551 | .528 | .398 | .557 | **.549** |
| Supervised | .964 | .978 | .963 | .915 | .912 | .979 | .877 | .956 |

*\*66.130 = oblique camera, unseen during (holdout-mode) training.*

### 5.7 Label efficiency

A legacy semi-supervised curve (ViT-S base, 2 seeds; Fig. 4) shows enrolling ~50% of identities matched full supervision (rank-1 0.835). Notably, the new *label-free* system's rank-5 (0.957) now exceeds that legacy 50%-enrolment rank-5 (0.932); re-running the curve on the current base is left as camera-ready work.

### 5.8 Heterogeneous backbone fusion: an animal-ReID foundation model

The twelve-family ablation (§5.5) concluded that *representation is not the bottleneck* — but every backbone tested there was a **natural-image** model (DINOv2, DINOv3, ImageNet GeM). We revisit this with **MegaDescriptor-L-384** [MegaDescriptor], a Swin-L foundation model trained specifically for *animal* re-identification on 29 wildlife/livestock datasets (its training corpus includes public cattle sets, but never our recording). Frozen, its features already beat frozen DINOv2 on exactly the sub-tasks where our system was weakest — dorsal-query mean 0.50 vs 0.42, full-transductive 0.46 vs 0.34 — confirming that the dorsal deficit is partly a *domain* gap of the backbone, not an information limit of the data.

Distilled through the identical deployment-mode recipe (only the last Swin stage + norm fine-tuned, 518→384 resize; ~6 GPU-min per 1000-step model), a MegaDescriptor student is weaker than a DINOv2 student on the oblique benchmark (P1 0.72 vs 0.90) but comparable-or-better on the dorsal cameras — the two backbones are **complementary**. Fusing their student ensembles at the distance level (Table 4) lifts *all three* protocols simultaneously, with **no human labels**: held-out-camera P1 **0.883 → 0.926**, dorsal-query mean **0.550 → 0.612**, full-transductive P2 **0.558 → 0.616**. This is the strongest label-free result on the dataset and, notably, the first intervention to break the dorsal plateau (0.51 → 0.61).

**Table 4 — Heterogeneous backbone fusion (label-free, all-camera deployment ensembles).**

| Ensemble | P1 (held-out) | Dorsal mean | P2 (transductive) |
|---|---|---|---|
| DINOv2 students only (6) | 0.883 | 0.550 | 0.558 |
| MegaDescriptor students only (3) | 0.718 | 0.556 | 0.509 |
| **DINOv2 + MegaDescriptor** | **0.926** | **0.612** | **0.616** |
| DINOv2 + MegaDescriptor (Mega ×2) | 0.914 | 0.625 | 0.611 |
| DINOv2 + MegaDescriptor + cluster rerank | 0.926 | 0.596 | **0.627** |

Two label-free operating points emerge from the same ensemble: maximal transductive accuracy (P2 0.627 with cluster reranking) or maximal dorsal-query accuracy (0.61–0.625 without). Takeaway: for uniform-coat, top-down livestock Re-ID, an *animal-domain* foundation model is not a drop-in replacement for natural-image SSL but a **complementary** one; the heterogeneous ensemble — not either backbone alone — is the lever, and it requires only off-the-shelf public weights.

### 5.9 Spatio-temporal mask

The impossibility mask is safe (0.37% of true matches wrongly excluded; zero queries losing all matches) but prunes only ~2.7% of gallery per query on P1, adding ≤ 0.004 mAP. Its value is expected to grow with denser temporal coverage in deployment.

## 6. Discussion and Limitations

**Scope of the headline.** 0.883 is a P1 (new-camera onboarding) number; averaged over query cameras the method is weaker than a supervised model everywhere except, paradoxically, the camera it never saw. Our gallery-ablation diagnostic (§5.6) shows the deficit is intrinsic to dorsal–dorsal matching of monochrome backs rather than contamination by the oblique gallery; closing it likely requires supervision-free sources of fine-grained dorsal cues (e.g., gait or body-contour dynamics across a tracklet) and is future work. **Meta-overfitting.** Design decisions were selected on P1; we mitigated with a-priori rationales, multi-seed validation, and the P2/P3 audit, but an untouched-camera confirmation would strengthen the claim. **Cost.** The trio needs three embedding passes at deployment (single-model distillation of the ensemble is untried). **Generality.** One barn, one session, one species; the transitivity-collapse law and the frozen-pairwise recipe are the components we expect to transfer, and they are the paper's core.

## 7. Conclusion

On sparse-overlap, look-alike livestock data, the bottleneck of unsupervised cross-camera Re-ID is not representation but the arithmetic of pseudo-label noise: transitive use of moderate-precision links is self-destructive, and single self-training runs are lotteries. Bounding both — proxies against merge damage, distance-ensembles against seed variance, frozen pairwise links against drift — recovers most of the supervised ceiling on the held-out-camera benchmark (rank-1 0.883, rank-5 0.957 vs 0.969) without a single identity label. We release code, per-protocol results, and the full negative-result ablation to make both the recipe and its limits reproducible.

---

## References

- [CAP] M. Wang, B. Lai, J. Huang, X. Gong, X.-S. Hua. *Camera-aware Proxies for Unsupervised Person Re-identification.* AAAI 2021. arXiv:2012.10674.
- [O2CAP] M. Wang, J. Li, B. Lai, X. Gong, X.-S. Hua. *Offline-Online Associated Camera-Aware Proxies for Unsupervised Person Re-Identification.* IEEE TIP 2022. doi:10.1109/TIP.2022.3213193.
- [ClusterContrast] Z. Dai, G. Wang, W. Yuan, S. Zhu, P. Tan. *Cluster Contrast for Unsupervised Person Re-identification.* ACCV 2022. arXiv:2103.11568.
- [SpCL] Y. Ge, F. Zhu, D. Chen, R. Zhao, H. Li. *Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.* NeurIPS 2020. arXiv:2006.02713.
- [IICS] S. Xuan, S. Zhang. *Intra-Inter Camera Similarity for Unsupervised Person Re-Identification.* CVPR 2021. arXiv:2103.11658.
- [CAJaccard] Y. Chen et al. *CA-Jaccard: Camera-aware Jaccard Distance for Person Re-identification.* CVPR 2024. arXiv:2311.10605.
- [PPLR] Y. Cho, W. J. Kim, S. Hong, S.-E. Yoon. *Part-based Pseudo Label Refinement for Unsupervised Person Re-identification.* CVPR 2022. arXiv:2203.14675.
- [TAUDL] M. Li, X. Zhu, S. Gong. *Unsupervised Person Re-identification by Deep Learning Tracklet Association.* ECCV 2018. arXiv:1809.02874.
- [TFusion] J. Lv, W. Chen, Q. Li, C. Yang. *Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns.* CVPR 2018. arXiv:1803.07293.
- [STReID] G. Wang, J. Lai, P. Huang, X. Xie. *Spatial-Temporal Person Re-identification.* AAAI 2019. arXiv:1812.03282.
- [MultiCamCows] P. Yu, T. Burghardt, A. W. Dowsey, N. W. Campbell. *Holstein-Friesian Re-Identification using Multiple Cameras and Self-Supervision on a Working Farm (MultiCamCows2024).* arXiv:2410.12695, 2024.
- [BNNeck] H. Luo, Y. Gu, X. Liao, S. Lai, W. Jiang. *Bag of Tricks and a Strong Baseline for Deep Person Re-Identification.* CVPR Workshops 2019. arXiv:1903.07071.
- [kReciprocal] Z. Zhong, L. Zheng, D. Cao, S. Li. *Re-ranking Person Re-identification with k-reciprocal Encoding.* CVPR 2017. arXiv:1701.08398.
- [RRF] G. V. Cormack, C. L. A. Clarke, S. Buettcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR 2009. doi:10.1145/1571941.1572114.
- [Market1501] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian. *Scalable Person Re-identification: A Benchmark.* ICCV 2015.
- [DINOv2] M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. arXiv:2304.07193.
- [GeM] F. Radenović, G. Tolias, O. Chum. *Fine-Tuning CNN Image Retrieval with No Human Annotation.* IEEE TPAMI 2019. arXiv:1711.02512.
- [SimCLR] T. Chen, S. Kornblith, M. Norouzi, G. Hinton. *A Simple Framework for Contrastive Learning of Visual Representations.* ICML 2020. arXiv:2002.05709.
- [Sinkhorn] M. Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.* NeurIPS 2013. arXiv:1306.0895.
- [AQE] O. Chum et al. *Total Recall: Automatic Query Expansion with a Generative Feature Model for Object Retrieval.* ICCV 2007. doi:10.1109/ICCV.2007.4408891.

## Figures

Fig. 1 `figures/fig1_journey_en_v1.png` · Fig. 2 `figures/fig2_ensemble_curve_en_v1.png` · Fig. 3 `figures/fig3_percamera_en_v1.png` · Fig. 4 `figures/fig4_label_efficiency_en_v1.png` · Fig. 5 `figures/fig5_methods_en_v1.png` · Fig. 6 `figures/fig6_pipeline_en_v3.png` · Fig. 7 `figures/fig7_mechanisms_en_v3.png`
