# stage1-cap

## Stage-1 overall flow (vitb_unsup_cap.py)  [vitb_unsup_cap.py:189-230]
Loop of --target steps (default 1000), wall-clock chunked and checkpoint-resumable. Every refresh_every=250 steps: (a) re-embed all training tracklets with the current model (256-d, T=2 evenly-spaced frames, eval mode); (b) inter-camera clustering = DBSCAN on CA-Jaccard distance over these embeddings with topology cannot-link pairs injected at distance 1.0; (c) embed all crop-bag crops with the current model and mine crop-level dustbin-OT must-links (min_conf=0.5, min_votes=3); (d) force-merge clusters connected by mined links via plain union-find; (e) rebuild the per-(cluster,camera) proxy memory from scratch from current embeddings. Training steps alternate: even step = intra-camera multi-branch cosine-classifier CE; odd step = CAP proxy loss (intra + inter camera) followed by momentum proxy update.

## (1) Intra-camera clustering algorithm  [vitb_unsup_cap.py:146-156; lib/cowreid/cluster.py:46-98]
Per camera, tracklet-mean frozen features are clustered by ClusterAssigner(0.7, 10): L2-normalize features; cosine sim matrix; build mutual-kNN edges (i,j) iff j is in i's top-k=10 AND i is in j's top-k=10 AND sim(i,j) >= 0.7; sort edges by descending similarity; add them to a CONSTRAINED union-find that refuses any union which would place a cannot-link pair in the same component (checks all cross pairs of the two components' member sets, lib/cowreid/pair_miner.py:347-361). min_cluster_size=1 so no cluster is discarded. Cluster count per camera (or 1 if zero) sets that camera's classifier size n_cls[c].

## (1) Intra-camera clustering features and constraint set  [vitb_unsup_cap.py:138-156]
Intra-camera clustering uses CACHED FROZEN DINOv2 ViT-S/14 features (dino_clip_feats_v1.npz, 384-d): frozen_mean[t] = mean over cached per-frame features, L2-normalized. The cannot-link set for intra clustering is restricted to SAME-CAMERA pairs only (cl_same = pairs whose two tracklets share a camera), i.e. same-camera time-overlapping tracklets can never merge. These intra labels are computed ONCE before training and NEVER refreshed.

## Cannot-link set construction  [lib/cowreid/cluster.py:21-43; vitb_unsup_cap.py:122-125]
build_cannot_link(tracklets, topology, overlap_threshold=0.02): a tracklet pair is cannot-link iff the two tracklets overlap in time AND (same camera OR their camera pair has topology overlap weight < 0.02). Semantics: a cow cannot be two tracklets in one camera at once, nor be in two non-overlapping locations at once. Tracklets are built with max_gap_s=2.

## Camera topology source (GT oracle)  [lib/cowreid/topology.py:47-63; vitb_unsup_cap.py:124-125]
CameraTopology.from_gt: overlap weight of a camera pair = fraction of co-occurring cross-camera crop pairs (same timestamp) that share a ground-truth identity. Pairs with weight >= 0.02 are 'overlapping'. This GT-derived topology feeds both the cannot-link set and the crop-bag construction. A label-free alternative (CameraTopology.estimate via dustbin-OT match rates) exists in the codebase but the Stage-1 scripts call from_gt.

## (2) CA-Jaccard distance (inter-camera clustering metric)  [lib/cowreid/cajaccard.py:20-65]
ca_jaccard_distance(feat, cameras, k1=20, k2=6, camera_aware=True, lambda_value=0.0): (i) L2-normalize, original distance d = clip(1 - cos, 0); (ii) k-reciprocal set R(i,k1): j in i's top-(k1+1) AND i in j's top-(k1+1); (iii) local expansion: for each j in R(i,k1) add R(j, round(k1/2)=10) if |R(j,10) ∩ R(i,20)| > (2/3)|R(j,10)|; (iv) CAMERA-AWARE step: append the max(1, k1//2)=10 nearest OTHER-camera samples by original distance; (v) row vector V_i: w = exp(-d(i,neighbors)) normalized to sum 1 over the neighbor set, zero elsewhere; (vi) local query expansion (k2=6): V_i <- mean of V over i's top-6 initial-rank neighbors (self included since diag(d)=0); (vii) Jaccard distance J(i,j) = 1 - sum_k min(V_ik,V_jk) / sum_k max(V_ik,V_jk); symmetrize J <- (J + J^T)/2; diagonal 0. lambda_value=0 so the original cosine distance is NOT mixed in.

## (2) DBSCAN on CA-Jaccard  [lib/cowreid/cajaccard.py:68-90; vitb_unsup_cap.py:114,184-185]
dbscan_cluster: compute CA-Jaccard matrix; set D[a,b]=D[b,a]=1.0 (max distance) for every cannot-link pair; sklearn DBSCAN(eps, min_samples=2, metric='precomputed').fit_predict. eps=0.5 in vitb_unsup_cap.py (arg default, line 114) — note train_finetune_iics.py uses eps=0.6. DBSCAN noise points (label -1) are converted to their own SINGLETON clusters, so every tracklet gets a cluster id (and hence a proxy).

## (3) Crop bags for OT mining  [lib/cowreid/crossview_ot.py:21-44; vitb_unsup_cap.py:105,134-136]
crossview_crop_bags: for every timestamp and every pair of cameras present at that timestamp whose pair is topology-overlapping (weight >= 0.02), form a bag (camA, camB, crop paths in A, crop paths in B), keeping only crops that belong to an indexed tracklet and cameras != holdout. Bags are randomly subsampled to max_bags=2000 (script arg; module default is 1500 but overridden).

## (3) Dustbin-OT matching (Sinkhorn with reject option)  [lib/cowreid/crossview_ot.py:62-81; lib/cowreid/sinkhorn.py:12-89]
For each bag: crop embeddings from the CURRENT model (each crop embedded as a T=1 clip, 256-d, L2-normalized); cost C = 1 - E_A E_B^T. match_with_dustbin augments C to (n+1)x(m+1): dustbin row/column filled with dustbin_cost = 0.5-quantile (median) of the real cost entries; corner aug[n,m]=0. Marginals: a = [1,...,1, m] and b = [1,...,1, n], each normalized to sum 1 (dustbin can absorb all mass of the other side). Solve entropic OT with log-domain Sinkhorn, eps=0.1, n_iters=200, tol=1e-6. Accept pair (i,j) iff (a) P[i,j] > P[i, dustbin] and (b) i = argmax over real rows of column j. Confidence = P[i,j] / (total row-i mass INCLUDING the dustbin column).

## (3) Vote aggregation into tracklet must-links  [lib/cowreid/crossview_ot.py:62-86; vitb_unsup_cap.py:196-197]
Each accepted crop match with confidence >= min_conf=0.5 casts one vote for the unordered tracklet pair {tracklet(cropA), tracklet(cropB)} (skipped if same tracklet); votes accumulate conf, counts accumulate 1, over ALL bags/timestamps. A tracklet pair becomes a must-link iff counts >= min_votes=3. GT is used only to LOG precision of the links, not to filter them.

## (3) Must-link merging  [train_finetune_iics.py:93-122; vitb_unsup_cap.py:198]
merge_labels(labels, must_links): plain (UNconstrained) union-find. First union all members of each existing DBSCAN cluster, then union each mined must-link pair, then relabel components 0..K-1. There is NO cannot-link check at this stage — a mined link can transitively merge clusters that contain cannot-link pairs.

## (4) Proxy definition and init  [vitb_unsup_cap.py:45-71,182-199]
A proxy is one vector per (inter-camera cluster id, camera) pair: all tracklets of cluster c observed in camera cam share one proxy. ProxyMemory.build: proxy = L2-normalized mean of the member tracklets' current model embeddings (256-d, computed by embed_tids with T=2 evenly spaced frames, eval mode, fp16 autocast then cast to fp32). Memory also precomputes, per proxy p: same_cam(p) = indices of all proxies whose camera equals cam(p); and pos_inter(p) = (diff, pos) where diff = proxies with camera != cam(p) and pos = subset of diff with the same cluster id as p. The memory is REBUILT from scratch (re-initialized from means) at every 250-step refresh; momentum state does not persist across refreshes.

## (4) Proxy momentum update rule  [vitb_unsup_cap.py:48,73-76,228]
After computing the CAP loss on a batch (and BEFORE optimizer.step()), for each instance feature f (detached, fp32, L2-normalized output of model.embed) with proxy index p, sequentially: proxy_p <- normalize(m * proxy_p + (1 - m) * f) with m = 0.2. Note the momentum coefficient 0.2 multiplies the OLD proxy, so each update is 80% new feature. Updates happen only on odd (CAP) steps; multiple instances of the same proxy in one batch compound sequentially.

## (5) CAP intra-camera loss (exact form)  [vitb_unsup_cap.py:79-88]
Let f_i be the L2-normalized 256-d embedding of instance i, pi(i) its proxy, tau = 0.07, and s_iq = f_i . p_q / tau the scaled cosine to proxy q. L_intra(i) = -log[ exp(s_{i,pi(i)}) / sum_{q in SameCam(pi(i))} exp(s_iq) ], i.e. softmax cross-entropy over ALL proxies in the instance's own camera (every cluster of that camera), positive = the instance's own proxy. Implemented via F.cross_entropy on the row restricted to same-camera proxy indices.

## (5) CAP inter-camera loss (exact form)  [vitb_unsup_cap.py:89-95]
With D_i = {q : cam(q) != cam(pi(i))} (ALL different-camera proxies, any cluster) and P_i = {q in D_i : cluster(q) = cluster(pi(i))} (different-camera proxies sharing the instance's cluster from the offline DBSCAN+OT association): L_inter(i) = -[ logsumexp_{q in P_i}(s_iq) - logsumexp_{q in D_i}(s_iq) ], same tau = 0.07. If P_i is empty (single-camera cluster), L_inter(i) = 0. This is a soft multi-positive pull toward all cross-camera proxies of the same cluster, normalized against all cross-camera proxies; it reduces to InfoNCE only when |P_i| = 1. There is no hard-mining / online top-k positive selection (offline-association CAP, not O2CAP's online part).

## (5) CAP total loss and weighting  [vitb_unsup_cap.py:96-97]
L_CAP = (1/B) sum_i [ L_intra(i) + L_inter(i) ] — intra and inter terms summed with EQUAL weight 1, mean over the batch; no separate weighting hyperparameter exists. Gradients flow only into f_i; proxies are a non-learnable memory (torch.no_grad update).

## (5) Intra-camera multi-branch CE branch (even steps)  [vitb_unsup_cap.py:203-215; lib/cowreid/iics.py:54-75]
On even steps: sample one camera c uniformly at random; batch from that camera's intra pseudo-labels; loss = CrossEntropy(16 * <f_i, w_hat_{c,y}>, y_i), where f_i is the L2-normalized 256-d embedding and w_hat are the L2-normalized rows of camera c's bias-free linear classifier (a cosine classifier with fixed scale 16.0). One independent classifier per camera (nn.ModuleDict), sized by that camera's intra cluster count.

## (5) Losses NOT present in the CAP script  [vitb_unsup_cap.py:203-228; train_phase2.py:33-45; lib/cowreid/losses.py:28-199; train_finetune_iics.py:260-277]
vitb_unsup_cap.py has exactly two loss branches: intra-camera cosine-classifier CE (even steps) and the CAP proxy loss (odd steps). There is NO ClusterMemory/ClusterNCE, no instance-level InfoNCE, and no explicit CannotLinkLoss in this script — those constitute the odd-step objective of the champion baselines (vitb_unsup.py and train_finetune_iics.py), which use build_objective: total = 1.0 * NegativeAwareContrastiveLoss(temp 0.07, hard negatives weighted x2 in the denominator) + 1.0 * ClusterNCE(q . centroids / 0.05, momentum-0.2 centroid memory, ignore_index=-1) + 0.5 * CannotLinkLoss = mean ReLU(cos(a,b) - 0) over in-batch cannot-link pairs.

## (6) P x K batch sampling  [vitb_unsup_cap.py:109-110,203-227]
P=12, K=4, batch = up to 48 tracklet-clips (each T=2 frames, so <= 96 frames of 518x518 per forward). Even (intra-CE) step: pick 1 camera uniformly; pick min(P, #labels) intra-cluster labels WITHOUT replacement; from each label pick K tracklets (WITH replacement iff the cluster has fewer than K members). Odd (CAP) step: pick min(P, #proxies) proxies uniformly WITHOUT replacement from ALL proxies (no camera or cluster balancing); from each proxy's member tracklets pick K with the same replacement rule; so all K instances of a group share one (cluster, camera) proxy.

## (7) Frames / clip construction  [vitb_unsup_cap.py:104,111,131; vitb_unsup.py:41,69-99; train_phase2_run.py:52-55]
Each tracklet has 8 cached frames (--frames 8) chosen by np.linspace over its full frame list (sample_frames), stored pre-resized to 518x518 uint8 in an mmap'd cache (_imgcache.npy) and normalized on GPU with ImageNet mean/std (0.485,0.456,0.406)/(0.229,0.224,0.225). At TRAIN time a clip is T=2 frames drawn uniformly WITH replacement from the 8 cached frames (rng.integers). At embedding/refresh time (train=False) T=2 frames evenly spaced by linspace over the 8 (padded by repeating the last if needed). No other data augmentation is applied in the CAP script.

## (7) Optimizer and schedule  [vitb_unsup_cap.py:106-108,161-164,190-232]
AdamW with two parameter groups: unfrozen backbone blocks at lr = 1e-5; the whole head (temporal pool + embed MLP + all per-camera classifiers) at lr = 3e-4; weight_decay = 1e-4 applied to both groups. No LR schedule / warmup / grad clipping. Mixed precision: torch.autocast fp16 + GradScaler; loss computed on emb cast to fp32 in the CAP branch. --target 1000 total steps (500 CE + 500 CAP by alternation), label refresh every 250 steps, run in wall-clock chunks (--wall 240 s) with checkpoint resume of model+optimizer+step.

## (8) Backbone and unfrozen layers  [vitb_unsup_cap.py:158-159; vitb_unsup.py:36; lib/cowreid/encoder.py:37-49]
timm 'vit_base_patch14_dinov2.lvd142m' (DINOv2 ViT-B/14, LVD-142M pretraining, embed_dim 768), input 518x518. requires_grad_(False) then unfreeze_last(4): only the LAST 4 transformer blocks plus the FINAL LayerNorm (model.norm) are trainable; all lower blocks and the patch embedding stay frozen but still run inside the autograd graph (no torch.no_grad).

## (8) Head architecture / embedding definition  [train_finetune_iics.py:35-54; lib/cowreid/iics.py:34-79; lib/cowreid/encoder.py:55-70]
FineTuneIICS.embed(clips): per-frame ViT-B features (B,T,768) -> MultiBranchReID.backbone = F.normalize( ReLU( AIBN1d( Linear(768 -> 256) ) ) applied after TemporalPool ). TemporalPool is learned attention: softmax over T of Linear(768,1) scores, weighted sum (NOT mean). AIBN1d: y = gamma * (alpha * BN(x) + (1-alpha) * InstanceStd(x)) + beta, alpha learnable clamped to [0,1] (init 0.5), BN affine=False, instance standardization is per-sample across the 256 dims. Output is the L2-normalized 256-d embedding used for proxies, clustering, OT mining, and evaluation.

## Training split  [vitb_unsup_cap.py:133,142; vitb_unsup.py:40]
Leave-one-camera-out: all tracklets from cameras != '66.130' (HOLD) are the unsupervised training set (loco_train); camera 66.130 is held out entirely from Stage-1 training and used as the query camera at evaluation.

## Init clustering vs step-0 refresh  [vitb_unsup_cap.py:179-199]
Before the loop, inter labels are computed once — on frozen ViT-S features if start_step==0, else on model embeddings E0 (resume path) — WITHOUT crop-OT merging, and the proxy memory is built from E0. But since start_step % 250 == 0 triggers the refresh at the first loop iteration, on a fresh run this init is immediately superseded at step 0 by DBSCAN on the (yet-untrained) model's own 256-d embeddings plus crop-OT merging. Effectively all training-time clusters and proxies live in the model's 256-d space; the frozen ViT-S features only determine the (fixed) intra-camera labels.

## train_finetune_iics.py tracklet-level mutual-NN mining (alternative to crop-OT)  [train_finetune_iics.py:56-90,207-241,300-301]
In train_finetune_iics.py (mine_mode='tracklet', its default), cross-view must-links come from tracklet-level mining instead of crop-OT: candidates = cross-camera tracklet pairs on a topology-overlapping camera pair that overlap in time; a candidate (a,b) is accepted iff cosine(emb_a, emb_b) >= sim_thr = 0.8 AND a and b are each other's best-scoring candidate partner (mutual-NN over the candidate graph). The CAP script always uses the crop-OT miner instead.

## train_finetune_iics odd-step masks  [train_phase2_run.py:77-88; train_finetune_iics.py:270-277]
make_masks(tids, inter_labels, cl): positives = same inter-cluster label (label >= 0); hard-negative mask and cannot_link_pairs = in-batch pairs present in the topology cannot-link set (the same pairs serve as both hard negatives weighted x2 in the InfoNCE denominator and as the hinge cannot-link pairs). forbid_negative_mask is NOT passed by the train loop (defaults to all-False).

## Where GT enters Stage 1  [vitb_unsup_cap.py:124-126,196-197; lib/cowreid/topology.py:47-63]
Ground-truth identities are used in exactly two non-loss places: (1) CameraTopology.from_gt (camera overlap weights, threshold 0.02) which feeds cannot-link constraints and crop-bag eligibility; (2) logging the precision of mined crop-OT links (gt passed to mine_crop_ot_links only for the printed prec). Pseudo-labels, proxies, and all losses are label-free.

## HYPERPARAMS
- Proxy/CAP temperature tau = 0.07   (vitb_unsup_cap.py:48,179)
- Proxy momentum m (multiplies OLD proxy) = 0.2  (update: p <- normalize(0.2 p + 0.8 f))   (vitb_unsup_cap.py:48,76)
- P (groups per batch) = 12   (vitb_unsup_cap.py:109)
- K (instances per group) = 4  (batch = 48 clips)   (vitb_unsup_cap.py:110)
- T (frames per clip at train and refresh-embed time) = 2   (vitb_unsup_cap.py:111)
- frames cached per tracklet (linspace-sampled) = 8   (vitb_unsup_cap.py:104)
- proj_dim (embedding dim) = 256   (vitb_unsup_cap.py:112)
- n_blocks unfrozen (ViT-B last blocks + final norm) = 4   (vitb_unsup_cap.py:113; lib/cowreid/encoder.py:37-48)
- DBSCAN eps on CA-Jaccard distance (CAP script) = 0.5   (vitb_unsup_cap.py:114)
- DBSCAN eps (train_finetune_iics.py) = 0.6   (train_finetune_iics.py:299)
- DBSCAN min_samples = 2 (noise points become singleton clusters)   (lib/cowreid/cajaccard.py:69,84-89)
- CA-Jaccard k1 (k-reciprocal) = 20 (local expansion uses round(k1/2)=10 with 2/3 overlap rule)   (lib/cowreid/cajaccard.py:27,41-42)
- CA-Jaccard k2 (local query expansion) = 6   (lib/cowreid/cajaccard.py:27,53-54)
- CA-Jaccard inter-camera neighbours added per sample = max(1, k1//2) = 10   (lib/cowreid/cajaccard.py:49)
- CA-Jaccard lambda (original-distance mixing) = 0.0 (pure Jaccard)   (lib/cowreid/cajaccard.py:28,62-63)
- Intra-camera ClusterAssigner sim_threshold = 0.7   (vitb_unsup_cap.py:152; lib/cowreid/cluster.py:49)
- Intra-camera ClusterAssigner k (mutual-kNN) = 10 (min_cluster_size=1)   (vitb_unsup_cap.py:152; lib/cowreid/cluster.py:49-50)
- Topology overlap threshold = 0.02 (cannot-link + crop-bag eligibility)   (vitb_unsup_cap.py:125; lib/cowreid/crossview_ot.py:21)
- Tracklet building max_gap_s = 2   (vitb_unsup_cap.py:122)
- Sinkhorn entropic eps (OT) = 0.1 (n_iters=200, tol=1e-6, log-domain)   (lib/cowreid/crossview_ot.py:62; lib/cowreid/sinkhorn.py:13)
- Dustbin cost = 0.5-quantile (median) of the bag's real cost entries; corner cell = 0   (lib/cowreid/sinkhorn.py:50,68-73)
- OT match min confidence (min_conf) = 0.5   (vitb_unsup_cap.py:197)
- OT link min votes (min_votes) = 3   (vitb_unsup_cap.py:197)
- max_bags (crop bags subsample) = 2000   (vitb_unsup_cap.py:105,134-135)
- Optimizer = AdamW; backbone lr 1e-5, head lr 3e-4; weight_decay 1e-4 (both groups); fp16 autocast + GradScaler; no LR schedule   (vitb_unsup_cap.py:161-164)
- Total steps (--target) = 1000 (alternating: 500 intra-CE + 500 CAP)   (vitb_unsup_cap.py:106,190,203-216)
- Label refresh interval = every 250 steps (re-cluster + re-mine + rebuild proxy memory)   (vitb_unsup_cap.py:108,191-199)
- Cosine-classifier logit scale = 16.0 (bias-free, L2-normalized weights and embeddings)   (lib/cowreid/iics.py:73-75)
- Backbone model = timm vit_base_patch14_dinov2.lvd142m (ViT-B/14 DINOv2), input 518x518, ImageNet normalization   (vitb_unsup.py:36,41; vitb_unsup_cap.py:158)
- Frozen-feature cache for intra labels = dino_clip_feats_v1.npz = frozen DINOv2 ViT-S/14 per-frame features, tracklet mean L2-normalized (384-d)   (vitb_unsup_cap.py:103,138-140)
- Holdout camera (excluded from Stage-1 training) = 66.130   (vitb_unsup.py:40; vitb_unsup_cap.py:133,142)
- Champion odd-step objective weights (train_finetune_iics/vitb_unsup, for contrast) = w_contrastive=1.0 (temp 0.07, hard-negative weight 2.0), w_cluster=1.0 (ClusterNCE temp 0.05, memory momentum 0.2), w_cannotlink=0.5 (hinge margin 0.0)   (train_phase2.py:33-45; lib/cowreid/losses.py)
- Tracklet mutual-NN must-link cosine threshold (train_finetune_iics only) = 0.8 (--crossview-sim)   (train_finetune_iics.py:300-301,73-90)
- Seed = 0 (torch, numpy, and all default_rng instances)   (vitb_unsup_cap.py:115,130-131,166)

## GOTCHAS
- Intra-camera pseudo-labels are FIXED for the whole run: computed once from cached frozen ViT-S/14 features (a different, smaller backbone than the ViT-B being trained) and never refreshed. Only the inter-camera clusters/proxies refresh every 250 steps. Do not describe intra labels as self-training.
- The proxy 'momentum 0.2' multiplies the OLD proxy: p <- normalize(0.2 p + 0.8 f). Each update is 80% new feature — the opposite of the slow-EMA convention (e.g. 0.9 on old). Writing 'momentum 0.2' without the formula will be misread.
- The CAP intra loss is a softmax over ALL proxies of the instance's camera (every cluster in that camera), not over the instance's cluster's proxies; the positive is the instance's own (cluster,camera) proxy.
- The CAP inter loss is NOT cross-entropy over clusters: it is -(logsumexp over same-cluster different-camera proxies minus logsumexp over ALL different-camera proxies), a multi-positive soft pull. It equals InfoNCE only when exactly one cross-camera proxy shares the cluster, and is exactly 0 for single-camera clusters. There is no online/top-k positive selection (offline CAP, not O2CAP's online association).
- Intra and inter CAP terms are summed with equal weight 1; there is no loss-weight hyperparameter to report.
- Crop-OT must-link merging (merge_labels) is a PLAIN union-find with no cannot-link check — topology constraints can be violated by mined merges. Cannot-links are enforced only (a) as hard max-distance entries in the DBSCAN matrix (still circumventable via density chains, since sklearn DBSCAN has no constraint mechanism) and (b) strictly, in the intra-camera constrained union-find.
- DBSCAN eps differs between scripts: 0.5 in vitb_unsup_cap.py vs 0.6 in train_finetune_iics.py — and eps is on the CA-Jaccard distance in [0,1], not on cosine distance.
- DBSCAN outliers are not discarded: each noise point becomes its own singleton cluster, so every tracklet owns a proxy and contributes to the losses (no -1/ignored labels in the CAP script).
- On a fresh run the frozen-feature init clustering (and the no-OT proxy build at lines 182-187) is immediately overwritten by the step-0 refresh, which clusters the untrained model's own 256-d embeddings and applies crop-OT merging. Training clusters therefore always live in the model's 256-d projected space, never in frozen ViT-S space.
- OT match confidence = P[i,j] / (row-i total mass INCLUDING the dustbin), and the dustbin cost is data-adaptive (median of the bag's costs), not a learned parameter as in SuperGlue. Marginals give the dustbin row/column mass m and n respectively before normalization.
- The camera topology used for cannot-links and crop bags is a GT ORACLE (CameraTopology.from_gt: fraction of co-occurring cross-camera crop pairs sharing an identity). Stage 1 is label-free in its losses, but the topology prior is derived from annotations (a label-free estimator exists in the codebase but is not used by these scripts); GT is also used to print mining precision.
- vitb_unsup_cap.py contains NO cannot-link hinge loss, NO ClusterNCE memory, and NO instance-level InfoNCE — those (weights 1/1/0.5, temps 0.07/0.05) belong to the champion baseline scripts (vitb_unsup.py, train_finetune_iics.py). Do not import their loss table into the CAP method description.
- The proxy memory update runs BEFORE optimizer.step() (with detached pre-step embeddings), only on odd steps, and updates sequentially per instance so K=4 same-proxy instances compound within one batch; the whole memory is re-initialized from cluster-camera means at every 250-step refresh, discarding momentum history.
- Batches are camera-homogeneous on even steps (one random camera per step) but proxy-uniform on odd steps (P=12 proxies drawn uniformly over all (cluster,camera) proxies, no camera balancing). 'P x K' refers to proxies (odd) or intra-clusters (even), not identities.
- Train-time clips are only T=2 frames sampled WITH replacement from 8 linspace-cached frames per tracklet (so duplicate frames within a clip are possible); refresh embeddings also use T=2 (evenly spaced), not all 8 frames. No image augmentation is applied (fixed 518x518 resize + ImageNet normalization only).
- unfreeze_last(4) also unfreezes the final LayerNorm (not just 4 blocks), and frozen lower blocks still run inside the autograd graph with requires_grad=False (no torch.no_grad wrapper).
- The 'embedding' is not a plain linear projection: TemporalPool is learned softmax attention over frames, followed by Linear(768->256) + AIBN1d (learnable BN/instance-norm mix, alpha clamped [0,1], init 0.5) + ReLU + L2 normalization. Cosine classifiers are bias-free with fixed scale 16.
- In train_finetune_iics.py's odd steps, the same in-batch cannot-link pairs serve double duty: as hard negatives (weight 2.0 in the InfoNCE denominator) and as the hinge cannot-link pairs (weight 0.5, margin 0); forbid_negative_mask is never passed by that training loop.