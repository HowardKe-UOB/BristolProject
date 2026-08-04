# Pairwise, Not Transitive: Label-Free Multi-Camera Cattle Re-Identification

Code for a fully **label-free** cross-camera cattle re-identification system:
7 ceiling cameras, 98 free-roaming cows, one 2.2-hour session, and **zero
identity annotations** during training. Rank-1 accuracy on a never-labelled
new camera rises from 0.503 (standard unsupervised baseline) to **0.945**,
within 2.4 points of a fully supervised model trained on the same backbone
(0.969).

The central measurement behind the design: cross-camera links mined at
52-60% precision collapse to ~17% pairwise precision once merged
transitively into clusters. The design rule follows: **pairwise, never
transitive** - every mined link trains exactly the pair it names and is
never chained into merged identities.

## Method in one paragraph

A four-stage "noise-bounded ensemble-distillation ladder", where every stage
limits the damage of pseudo-label errors: (1) self-training with
camera-aware proxies (a wrong merge corrupts one proxy, not a cluster
centroid); (2) a teacher built by averaging five independently seeded
models (cancels seed luck; mining in this space is 4x cleaner); (3)
frozen-label pairwise-link distillation into student models (no refresh, no
feedback loop; links act as confidence-weighted pairs); (4) heterogeneous
fusion of DINOv2 students with MegaDescriptor students (the two backbone
families fail on different animals and correct each other). A label-free
inference stack (768-d read-out, flip TTA, per-camera centering, whitening,
re-ranking, rank fusion) adds ~21 points without touching any weights.

## Data

The `2025Sep18` recording (124,145 body crops from 7 cameras) was collected
and preprocessed by **Phoenix Yu** (University of Bristol): SAM-3
segmentation, tracking by his modified gSORT, cross-camera identities
manually linked (used for evaluation only). The dataset is **not included**
in this repository; contact the data author. Pretrained backbones: DINOv2
(Meta AI), MegaDescriptor (BVRA). Everything downstream - tracklet
formation, free-signal mining, topology analysis, the training ladder, and
all evaluation - is this repository.

## Repository map

| Path | Contents |
|---|---|
| `cowreid/` | Library: manifest/tracklets/topology (data), pair mining + OT + constrained clustering (signals), encoder/losses/batching (training), splits/eval/ST-mask (evaluation) |
| `build_phase1.py` | Phase-1 driver: manifest -> tracklets -> mined signals -> splits |
| `vitb_unsup.py` | Core trainer infrastructure (image cache, chunked checkpoint-resume) |
| `vitb_unsup_cap.py` | Stage 1: camera-aware proxy self-training (run with 5 seeds) |
| `eval_cap_ensemble.py` | Stage 2: five-seed distance-ensemble teacher |
| `make_distill_labels.py` | Stage 2->3: frozen label set (intra clusters + pairwise links) |
| `vitb_unsup_distill.py`, `vitb_unsup_deploy.py`, `vitb_unsup_hardcl2.py` | Stage 3: distilled students (holdout / deployment / hard cannot-link) |
| `vitb_unsup_mega.py` | Stage 4: MegaDescriptor students |
| `fuse_hetero.py`, `fuse_final.py`, `ensemble_search.py` | Stage 4: heterogeneous fusion and final operating points |
| `eval_sweep.py` | Three-protocol evaluation (also home of `champ_dist`, imported widely) |
| `artifacts2/` | 96 archived result JSONs - every number in the dissertation traces to one of these |
| `docs/reference/` | Per-stage fact sheets with file:line citations |
| remaining scripts | Diagnostics, ablations, negative-result experiments (14 failed families), figure generation |

## Reproducing the pipeline

```
# 0. place 2025Sep18.tar.gz + 2025Sep18.listing.txt in the repo root
pip install -r requirements.txt

# 1. signals and splits
python build_phase1.py

# 2. stage 1: five CAP seeds (GPU, ~40 min each)
python vitb_unsup_cap.py --seed 0 --ckpt _vitb_cap_s0_ckpt.pt   # ... repeat for seeds 1-4

# 3. stage 2: ensemble teacher + frozen labels
python eval_cap_ensemble.py
python make_distill_labels.py

# 4. stage 3: students (3 seeds each)
python vitb_unsup_deploy.py ...
python vitb_unsup_hardcl2.py ...

# 5. stage 4: MegaDescriptor students + fusion
python vitb_unsup_mega.py ...
python fuse_final.py

# 6. evaluation
python eval_sweep.py ...
```

Expect results within the reported seed variance (students +-0.03);
bit-exact reproduction of the dissertation's numbers requires the original
checkpoints (48 checkpoints, ~51 GB - available on request). All fusion and
evaluation steps run on CPU from saved embeddings.

## Citation

Dissertation: *Pairwise, Not Transitive: Noise-Bounded Ensemble Distillation
for Label-Free Multi-Camera Cattle Re-Identification*, University of
Bristol, 2026. Related dataset lineage: Yu et al., *Computers and
Electronics in Agriculture* 2025 (MultiCamCows2024); Yu et al., arXiv
2602.15962 (DazzleCows).

## License

MIT (see LICENSE). Note: MegaDescriptor pretrained weights carry their own
licence terms - check the BVRA release before commercial use.
